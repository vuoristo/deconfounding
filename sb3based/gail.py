import argparse
import collections
import json
import os
import sys
import tempfile
import pickle

import gym
import numpy as np
import rl_zoo3.import_envs  # noqa: F401 pylint: disable=unused-import
import stable_baselines3.common.logger as sb_logger
import torch as th
import tree
import yaml
from huggingface_sb3 import EnvironmentName
from rl_zoo3 import ALGOS, create_test_env, get_saved_hyperparams
from rl_zoo3.exp_manager import ExperimentManager
from rl_zoo3.utils import StoreDict, get_model_path
from stable_baselines3.common.callbacks import tqdm
from stable_baselines3.common.utils import set_random_seed
from ray import tune
from ray.air import session, RunConfig
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy

from sb3based.rnnpolicy import RewardNet, KeyCombinedExtractor
from sb3based import reward_wrapper
from sb3based.ortega_bandit import BanditEnv
from sb3based.rnnbc import (
    TimeStep,
    ReplayBuffer,
    pack_structure,
    get_expert_sample_fn,
    get_ortega_expert_sample_fn,
)


from gym import Wrapper
class ObsActionWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        spaces = env.observation_space.spaces
        spaces["action_tm1"] = env.action_space
        self.observation_space = gym.spaces.Dict(spaces)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        obs["action_tm1"] = np.zeros_like(self.env.action_space.sample())
        return obs

    def step(self, action):
        o, r, d, i = self.env.step(action)
        o["action_tm1"] = np.array(action)
        return o, r, d, i


class IdentityWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)


DiscLog = collections.namedtuple(
    "DiscLog", ["loss", "disc_accuracy", "grad_norm"]
)
PiLog = collections.namedtuple("PiLog", ["neglogp", "entropy", "grad_norm"])


def get_bc_sample_fn(model, action_space, is_vectorized):
    def sample_fn(obs, state, action_tm1, episode_start, deterministic, inject_noise_probability, frac=1.0):
        with th.no_grad():
            action_tm1 = th.tensor(action_tm1, device=model.device)
            episode_start = th.tensor(episode_start, device=model.device).float()
            obs_t = tree.map_structure(
                lambda x: th.tensor(x, device=model.device), obs
            )
            actions, _, _, state = model(obs_t, state, episode_start, deterministic)
            actions = actions.detach().cpu().numpy().reshape((-1,) + action_space.shape)
        if not is_vectorized:
            actions = actions[0]
        return actions, actions, state

    return sample_fn


def sample_data(
    env,
    sample_fn,
    rnn_resetter,
    stochastic,
    deterministic,
    n_episodes,
    max_episode_length,
    progress,
    is_atari,
    n_envs,
    schedule_timesteps,
    timesteps_so_far,
    inject_noise_probability=0.0,
):
    obs = env.reset()
    rnn_state = rnn_resetter(n_envs)
    timesteps = [
        [
            TimeStep(
                tree.map_structure(lambda x: x[i], obs),
                np.zeros_like(env.action_space.sample()),
                np.zeros_like(env.action_space.sample()),
                np.zeros(1)[0],
                np.zeros(1)[0],
            )
        ]
        for i in range(obs["obs"].shape[0])
    ]
    action = np.tile(np.zeros_like(env.action_space.sample()), (n_envs, 1))
    episodes = []

    # Deterministic by default except for atari games
    stochastic = stochastic or is_atari and not deterministic
    deterministic = not stochastic

    ep_rews, ep_lens = [], []
    ep_rew = np.zeros((env.num_envs,))
    ep_len = np.zeros((env.num_envs,), dtype=int)
    episode_start = np.ones((env.num_envs,), dtype=bool)

    while len(episodes) < n_episodes:
        timesteps_so_far += n_envs
        frac = 1.0 - max(0, schedule_timesteps - timesteps_so_far) / schedule_timesteps
        action, expert_action, rnn_state = sample_fn(
            obs,  # type: ignore[arg-type]
            state=rnn_state,
            action_tm1=action,
            episode_start=episode_start,
            deterministic=deterministic,
            inject_noise_probability=inject_noise_probability,
            frac=frac,
        )
        obs, reward, done, infos = env.step(action)

        episode_start = done
        ep_len += 1
        ep_rew += np.array([i["original_env_rew"] for i in infos])

        for i in range(n_envs):
            if done[i] or ep_len[i] >= max_episode_length:
                last_obs = infos[i].get(
                    "terminal_observation", tree.map_structure(lambda x: x[i], obs)
                )
                tstep = TimeStep(last_obs, action[i], action[i], reward[i], 1.0)
                timesteps[i].append(tstep)
                episodes.append(pack_structure(timesteps[i]))
                timesteps[i] = [
                    TimeStep(
                        tree.map_structure(lambda x: x[i], obs),
                        np.zeros_like(env.action_space.sample()),
                        np.zeros_like(env.action_space.sample()),
                        np.zeros(1)[0],
                        np.zeros(1)[0],
                    )
                ]
                ep_lens.append(ep_len[i])
                ep_len[i] = 0
                if not is_atari:
                    ep_rews.append(ep_rew[i])
                    ep_rew[i] = 0
            else:
                tstep = TimeStep(
                    tree.map_structure(lambda x: x[i], obs),
                    action[i],
                    expert_action[i],
                    reward[i],
                    done[i],
                )
                timesteps[i].append(tstep)

        for info in infos:
            if is_atari and infos is not None:
                episode_infos = info.get("episode")
                if episode_infos is not None:
                    ep_rews.append(episode_infos["r"])

    return episodes, ep_rews, ep_lens, timesteps_so_far


def discriminator_update(
    imitation_env,
    bc_sample_fn,
    expert_sample_fn,
    bcmodel,
    rnn_resetter,
    reward_net,
    optimizer,
    expert_buffer,
    policy_buffer,
    stochastic,
    n_train_steps,
    n_episodes,
    max_episode_length,
    is_atari,
    n_envs,
    grad_clip,
    schedule_timesteps,
    timesteps_so_far,
    should_train,
    logger,
    inject_noise_probability,
):
    expert_steps, expert_rewards, expert_lens, timesteps_so_far = sample_data(
        imitation_env,
        expert_sample_fn,
        rnn_resetter,
        stochastic,
        False,
        n_episodes,
        max_episode_length,
        False,
        is_atari,
        n_envs,
        schedule_timesteps=schedule_timesteps,
        timesteps_so_far=timesteps_so_far,
        inject_noise_probability=inject_noise_probability,
    )
    for ep in expert_steps:
        expert_buffer.add_episode(ep)
    bc_steps, bc_rewards, bc_lens, _ = sample_data(
        imitation_env,
        bc_sample_fn,
        rnn_resetter,
        stochastic,
        False,
        n_episodes,
        max_episode_length,
        False,
        is_atari,
        n_envs,
        schedule_timesteps=schedule_timesteps,
        timesteps_so_far=timesteps_so_far,
        inject_noise_probability=0.0,
    )
    for ep in bc_steps:
        policy_buffer.add_episode(ep)

    if not should_train:
        return timesteps_so_far

    device = bcmodel.device
    reward_net.train()
    bcmodel.policy.eval()
    logs = []
    for i in range(n_train_steps):
        raw_batch = expert_buffer.sample(n_episodes, max_episode_length+1)._asdict()
        expert_batch = {}
        expert_batch["obs"] = tree.map_structure(lambda x: x[:-1], raw_batch["obs"])
        expert_batch["next_obs"] = tree.map_structure(lambda x: x[1:], raw_batch["obs"])
        expert_batch["action"] = raw_batch["action_tm1"][1:]
        expert_batch["expert_action"] = raw_batch["expert_action_tm1"][1:]
        expert_batch["done"] = raw_batch["done"][1:]
        raw_batch = policy_buffer.sample(n_episodes, max_episode_length+1)._asdict()
        policy_batch = {}
        policy_batch["obs"] = tree.map_structure(lambda x: x[:-1], raw_batch["obs"])
        policy_batch["next_obs"] = tree.map_structure(lambda x: x[1:], raw_batch["obs"])
        policy_batch["action"] = raw_batch["action_tm1"][1:]
        policy_batch["expert_action"] = raw_batch["expert_action_tm1"][1:]
        policy_batch["done"] = raw_batch["done"][1:]
        batch = tree.map_structure(lambda a, b: np.concatenate([a, b], axis=0), expert_batch, policy_batch)
        batch = tree.map_structure(lambda x: th.tensor(x, device=device), batch)
        T, B = policy_batch["done"].shape
        labels = th.cat([th.ones(T, B), th.zeros(T, B)], dim=0).to(device)
        logits = reward_net(batch["obs"], batch["expert_action"], batch["next_obs"], batch["done"]).squeeze(-1)
        loss = th.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip is not None:
            th.nn.utils.clip_grad_norm_(reward_net.parameters(), grad_clip)
        optimizer.step()
        total_norm = 0.0
        for p in reward_net.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm**2
        total_norm = total_norm**0.5
        with th.no_grad():
            # Logits of the discriminator output; >0 for expert samples, <0 for generator.
            bin_is_generated_pred = logits < 0
            # Binary label, so 1 is for expert, 0 is for generator.
            bin_is_generated_true = labels == 0
            correct_vec = th.eq(bin_is_generated_pred, bin_is_generated_true)
            acc = th.mean(correct_vec.float())
        disclog = tree.map_structure(
            lambda x: x.detach().cpu().numpy(),
            DiscLog(loss, acc, total_norm),
        )
        logs.append(disclog)
    disclog = tree.map_structure(np.mean, pack_structure(logs))
    for k, v in disclog._asdict().items():
        logger.record(f"disc/{k}", v)
    logger.record("bc/rewards", np.mean(bc_rewards))
    logger.record("bc/lens", np.mean(bc_lens))
    logger.record("expert/rewards", np.mean(expert_rewards))
    logger.record("expert/lens", np.mean(expert_lens))
    return timesteps_so_far


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="ray or local", type=str, default="local")
    parser.add_argument("--env-name", help="environment ID", type=str, default="OrtegaBandit-v0")
    parser.add_argument("--imitation-env-name", help="environment ID", type=str)
    parser.add_argument("--imitator-feature-extractor-obs-keys", nargs="+", type=str, default=["obs", "action_tm1"])
    parser.add_argument(
        "-f", "--folder", help="Log folder", type=str, default="rl-trained-agents"
    )
    parser.add_argument(
        "--algo",
        help="RL Algorithm",
        default="ppo",
        type=str,
        required=False,
        choices=list(ALGOS.keys()),
    )
    parser.add_argument(
        "-n", "--n-episodes", help="number of episodes", default=10, type=int
    )
    parser.add_argument(
        "--max-episode-length", help="max length of episodes", default=1000, type=int
    )
    parser.add_argument(
        "--num-threads",
        help="Number of threads for PyTorch (-1 to use default)",
        default=-1,
        type=int,
    )
    parser.add_argument("--n-envs", help="number of environments", default=1, type=int)
    parser.add_argument(
        "--exp-id",
        help="Experiment ID (default: 0: latest, -1: no exp folder)",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=False,
        help="Use deterministic actions",
    )
    parser.add_argument(
        "--device",
        help="PyTorch device to be use (ex: cpu, cuda...)",
        default="auto",
        type=str,
    )
    parser.add_argument(
        "--load-best",
        action="store_true",
        default=False,
        help="Load best model instead of last model if available",
    )
    parser.add_argument(
        "--load-checkpoint",
        type=int,
        help="Load checkpoint instead of last model if available, "
        "you must pass the number of timesteps corresponding to it",
    )
    parser.add_argument(
        "--load-last-checkpoint",
        action="store_true",
        default=False,
        help="Load last checkpoint instead of last model if available",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        default=False,
        help="Use stochastic actions",
    )
    parser.add_argument(
        "--norm-reward",
        action="store_true",
        default=False,
        help="Normalize reward if applicable (trained with VecNormalize)",
    )
    parser.add_argument(
        "--schedule-timesteps",
        help="dagger schedule timesteps",
        default=1000000,
        type=int,
    )
    parser.add_argument("--disc-lr", help="disc lr", default=1e-3, type=float)
    parser.add_argument("--disc-grad-clip", help="disc gradient clipping", type=float)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--disc-train-steps", type=int, default=1)
    parser.add_argument("--inject-noise-probability", help="injects noise to expert actions at this probability", default=0.0, type=float)
    parser.add_argument(
        "--buffer-size",
        help="how many trajectories to keep in memory",
        default=1000,
        type=int,
    )
    parser.add_argument("--imitator-gamma", default=0.99, type=float)
    parser.add_argument("--imitator-gae-lambda", default=0.95, type=float)
    parser.add_argument("--imitator-learning-rate", default=3e-4, type=float)
    parser.add_argument("--imitator-batch-size", default=None, type=int)
    parser.add_argument("--imitator-n-epochs", default=10, type=int)
    parser.add_argument("--imitator-max-grad-norm", default=0.5, type=float)
    parser.add_argument("--imitator-train-steps", default=1, type=int)
    parser.add_argument("--imitator-normalize-advantage", action="store_true", default=False)
    parser.add_argument("--reward-net-inputs", nargs="+", type=str, default=["state", "action", "next_state"])
    parser.add_argument("--reward-net-obs-keys", nargs="+", type=str, default=["obs"])
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--traj-save-interval", type=int, default=0)
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    parser.add_argument("--logdir", help="Where to log", default="", type=str)
    args = parser.parse_args()
    return args


def save_trajs(trajs, path, rlalgo, rnn_resetter):
    device = rlalgo.device
    # pi.get_distribution swaps T, B internally so we pack along axis 0
    trajs = pack_structure(trajs, axis=0)
    B, T = trajs.done.shape[:2]
    with th.no_grad():
        lstm_states = rnn_resetter(B)
        episode_starts = th.cat([th.ones(B, 1), th.zeros(B, T-1)], dim=1).to(device)
        pi = rlalgo.policy
        obs = pi.obs_to_tensor(tree.map_structure(lambda x: x.reshape(-1, *x.shape[2:]), trajs.obs))[0]
        features = pi.extract_features(obs)
        latent_pi, _ = pi._process_sequence(features, lstm_states.pi, episode_starts, pi.lstm_actor)
        latent_pi = pi.mlp_extractor.forward_actor(latent_pi)
        distributions = pi._get_action_dist_from_latent(latent_pi)
        logits = distributions.distribution.logits.reshape(B, T, -1).detach().cpu().numpy()
    trajs = tree.map_structure(lambda x: np.swapaxes(x, 1, 0), trajs)
    trajs = dict(trajs._asdict())
    trajs['logits'] = logits.swapaxes(1, 0)
    with open(path, 'wb') as f:
        pickle.dump(trajs, f)


def get_rnn_resetter(rlalgo):
    def rnn_resetter(B):
        lstm = rlalgo.policy.lstm_actor
        single_hidden_state_shape = (lstm.num_layers, B, lstm.hidden_size)
        # hidden and cell states for actor and critic
        states = RNNStates(
            (
                th.zeros(single_hidden_state_shape, device=rlalgo.device),
                th.zeros(single_hidden_state_shape, device=rlalgo.device),
            ),
            (
                th.zeros(single_hidden_state_shape, device=rlalgo.device),
                th.zeros(single_hidden_state_shape, device=rlalgo.device),
            ),
        )
        return states
    return rnn_resetter


def enjoy(
    algo,
    buffer_size,
    schedule_timesteps,
    deterministic,
    device,
    disc_lr,
    disc_grad_clip,
    disc_train_steps,
    env_name,
    imitation_env_name,
    epochs,
    exp_id,
    folder,
    imitator_feature_extractor_obs_keys,
    imitator_gamma,
    imitator_gae_lambda,
    imitator_learning_rate,
    imitator_batch_size,
    imitator_n_epochs,
    imitator_max_grad_norm,
    imitator_train_steps,
    imitator_normalize_advantage,
    reward_net_inputs,
    reward_net_obs_keys,
    load_best,
    load_checkpoint,
    load_last_checkpoint,
    logdir,
    n_envs,
    n_episodes,
    max_episode_length,
    norm_reward,
    num_threads,
    seed,
    stochastic,
    warmup_epochs,
    traj_save_interval,
    inject_noise_probability,
    *args,
    is_ray=False,
    **kwargs,
):
    logger = sb_logger.configure(os.path.join(logdir, "train"), ["stdout", "csv"])
    eval_logger = sb_logger.configure(os.path.join(logdir, "eval"), ["stdout", "csv"])
    rl_logger = sb_logger.configure(os.path.join(logdir, "rlalgo"), ["stdout", "csv"])

    env_name = EnvironmentName(env_name)
    algo = algo
    folder = folder

    if env_name.gym_id == "OrtegaBandit-v0":
        hyperparams = {}
        hyperparams["env_wrapper"] = ["sb3based.gail_wrappers.ObsActionWrapper"]
        maybe_stats_path = None
        log_path = ""
    else:
        _, model_path, log_path = get_model_path(
            exp_id,
            folder,
            algo,
            env_name,
            load_best,
            load_checkpoint,
            load_last_checkpoint,
        )
        stats_path = os.path.join(log_path, env_name)
        hyperparams, maybe_stats_path = get_saved_hyperparams(
            stats_path, norm_reward=norm_reward, test_mode=True
        )
        print(f"Loading {model_path}")


    # Off-policy algorithm only support one env for now
    off_policy_algos = ["qrdqn", "dqn", "ddpg", "sac", "her", "td3", "tqc"]

    if algo in off_policy_algos:
        n_envs = 1

    set_random_seed(seed)

    if num_threads > 0:
        th.set_num_threads(num_threads)

    is_atari = ExperimentManager.is_atari(env_name.gym_id)

    # load env_kwargs if existing
    env_kwargs = {}
    args_path = os.path.join(log_path, env_name, "yml")
    if os.path.isfile(args_path):
        with open(args_path) as f:
            loaded_args = yaml.load(
                f, Loader=yaml.UnsafeLoader
            )  # pytype: disable=module-attr
            if loaded_args["env_kwargs"] is not None:
                env_kwargs = loaded_args["env_kwargs"]

    log_dir = logdir if logdir != "" else None

    if is_atari:
        # Fix https://github.com/DLR-RM/stable-baselines3/issues/1060
        hyperparams["env_wrapper"] = ["sb3based.atari_wrapper.AtariWrapper"]

    # Simple copy here to avoid modifying the hyperparams loaded from the saved model
    hparams = hyperparams.copy()

    if imitation_env_name is not None:
        env_name = EnvironmentName(imitation_env_name)

    imitation_env = create_test_env(
        env_name.gym_id,
        n_envs=n_envs,
        stats_path=maybe_stats_path,
        seed=seed,
        log_dir=log_dir,
        should_render=False,
        hyperparams=hparams,
        env_kwargs=env_kwargs,
    )

    kwargs = dict(seed=seed)
    if algo in off_policy_algos:
        # Dummy buffer size as we don't need memory to enjoy the trained agent
        kwargs.update(dict(buffer_size=1))
        # Hack due to breaking change in v1.6
        # handle_timeout_termination cannot be at the same time
        # with optimize_memory_usage
        if "optimize_memory_usage" in hyperparams:
            kwargs.update(optimize_memory_usage=False)

    # Check if we are running python 3.8+
    # we need to patch saved model under python 3.6/3.7 to load them
    newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8

    custom_objects = {}
    if newer_python_version:
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }

    if env_name.gym_id == "OrtegaBandit-v0":
        expert_sample_fn = get_ortega_expert_sample_fn(imitation_env.action_space, 0.4)
        feature_extractor_kwargs = {"obs_keys": ["obs"]}
        feature_extractor_class = KeyCombinedExtractor
    else:
        expert_model = ALGOS[algo].load(
            model_path,
            env=imitation_env,
            custom_objects=custom_objects,
            device=device,
            **kwargs,
        )
        expert_sample_fn = get_expert_sample_fn(expert_model, imitation_env.action_space)
        feature_extractor_kwargs = expert_model.policy.features_extractor_kwargs.copy()
        feature_extractor_class = expert_model.policy.features_extractor_class
    feature_extractor_kwargs["obs_keys"] = reward_net_obs_keys
    reward_net = RewardNet(
        imitation_env.observation_space,
        imitation_env.action_space,
        features_extractor_class=feature_extractor_class,
        features_extractor_kwargs=feature_extractor_kwargs,
        inputs=reward_net_inputs,
    )
    reward_net.to(device)
    imitation_env = reward_wrapper.RewardVecEnvWrapper(
        imitation_env,
        reward_fn=reward_net.reward,
    )
    disc_optimizer = th.optim.AdamW(reward_net.parameters(), lr=disc_lr)
    feature_extractor_kwargs["obs_keys"] = imitator_feature_extractor_obs_keys
    rlalgo = RecurrentPPO(
        policy=RecurrentActorCriticPolicy,
        env=imitation_env,
        verbose=0,
        seed=seed,
        device=device,
        n_steps=max_episode_length,
        policy_kwargs=dict(
            features_extractor_class=feature_extractor_class,
            features_extractor_kwargs=feature_extractor_kwargs,
        ),
        gamma=imitator_gamma,
        gae_lambda=imitator_gae_lambda,
        learning_rate=imitator_learning_rate,
        batch_size=imitator_batch_size,
        n_epochs=imitator_n_epochs,
        max_grad_norm=imitator_max_grad_norm,
        normalize_advantage=imitator_normalize_advantage,
    )
    rlalgo.set_logger(rl_logger)
    rnn_resetter = get_rnn_resetter(rlalgo)
    example_timestep = TimeStep(
        tree.map_structure(lambda t: t[0], imitation_env.reset()),
        np.zeros_like(imitation_env.action_space.sample()),
        np.zeros_like(imitation_env.action_space.sample()),
        np.zeros(1)[0],
        np.zeros(1)[0],
    )
    expert_buffer = ReplayBuffer(buffer_size, example_timestep)
    policy_buffer = ReplayBuffer(buffer_size, example_timestep)
    bc_sample_fn = get_bc_sample_fn(rlalgo.policy, imitation_env.action_space, True)
    timesteps_so_far = 0
    for i in range(epochs):
        should_train = i >= warmup_epochs
        if should_train:
            rlalgo.policy.eval()
            trajs, rewards, lens, _ = sample_data(
                imitation_env,
                bc_sample_fn,
                rnn_resetter,
                stochastic,
                deterministic,
                n_episodes,
                max_episode_length,
                False,
                is_atari,
                n_envs,
                schedule_timesteps=schedule_timesteps,
                timesteps_so_far=timesteps_so_far,
            )
            r = np.mean(rewards)
            l = np.mean(lens)
            eval_logger.record("bc/rewards", r)
            eval_logger.record("bc/lens", l)
            eval_logger.dump()
            if is_ray:
                session.report({"rewards": r, "lens": l})
            if i % 5000 == 0 or i == epochs - 1:
                rlalgo.save(os.path.join(log_dir, f"rlalgo_{i}"))
            if traj_save_interval > 0 and (i % traj_save_interval == 0 or i == epochs - 1):
                save_trajs(trajs, os.path.join(log_dir, f"trajs_{i}.pkl"), rlalgo, rnn_resetter)
                expert_trajs, rewards, lens, _ = sample_data(
                    imitation_env,
                    expert_sample_fn,
                    rnn_resetter,
                    stochastic,
                    deterministic,
                    n_episodes,
                    max_episode_length,
                    False,
                    is_atari,
                    n_envs,
                    schedule_timesteps=schedule_timesteps,
                    timesteps_so_far=timesteps_so_far,
                )
                save_trajs(expert_trajs, os.path.join(log_dir, f"expert_trajs_{i}.pkl"), rlalgo, rnn_resetter)
        new_timesteps = discriminator_update(
            imitation_env=imitation_env,
            bc_sample_fn=bc_sample_fn,
            expert_sample_fn=expert_sample_fn,
            bcmodel=rlalgo,
            rnn_resetter=rnn_resetter,
            reward_net=reward_net,
            optimizer=disc_optimizer,
            expert_buffer=expert_buffer,
            policy_buffer=policy_buffer,
            stochastic=stochastic,
            n_train_steps=disc_train_steps,
            n_episodes=n_episodes,
            max_episode_length=max_episode_length,
            is_atari=is_atari,
            n_envs=n_envs,
            grad_clip=disc_grad_clip,
            schedule_timesteps=schedule_timesteps,
            timesteps_so_far=timesteps_so_far,
            should_train=should_train,
            logger=logger,
            inject_noise_probability=inject_noise_probability
        )
        if should_train:
            rlalgo.learn(
                total_timesteps=n_episodes * max_episode_length,
                reset_num_timesteps=False,
            )
        timesteps_so_far = new_timesteps
        logger.record("train/step", i)
        logger.record("train/expert_timesteps", timesteps_so_far)
        logger.dump()

    imitation_env.close()
    return rlalgo



def trainable(config):
    config["logdir"] = os.getcwd()
    with open(os.path.join(config["logdir"], "args.json"), "w") as f:
        json.dump(config, f)
    enjoy(**config, is_ray=True)


if __name__ == '__main__':
    args = get_args()
    if args.mode == 'ray':
        trial_space = args.__dict__
        trial_space.update({
            "kl_weight": tune.grid_search([0.0, 1.0]),
        })
        trainable = tune.with_resources(trainable, {"gpu": 1})
        tuner = tune.Tuner(trainable, param_space=trial_space, run_config=RunConfig(local_dir=args.logdir))
        tuner.fit()
    else:
        log_to_temp = len(args.logdir) == 0
        if log_to_temp:
            logdir = tempfile.mkdtemp()
        else:
            logdir = args.logdir
            os.makedirs(logdir, exist_ok=False)
        with open(os.path.join(logdir, "args.json"), "w") as f:
            json.dump(args.__dict__, f)
        print(f"Logging to {logdir}")
        args.logdir = logdir
        enjoy(**vars(args), is_ray=False)
