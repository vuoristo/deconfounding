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
from gym import Wrapper
from huggingface_sb3 import EnvironmentName
from rl_zoo3 import ALGOS, create_test_env, get_saved_hyperparams
from rl_zoo3.exp_manager import ExperimentManager
from rl_zoo3.utils import StoreDict, get_model_path
from stable_baselines3.common.callbacks import tqdm
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.utils import set_random_seed
from ray import tune
from ray.air import session, RunConfig

from sb3based.rnnpolicy import DeconfoundedRNNPolicy, RNNPolicy, KeyCombinedExtractor
from sb3based.ortega_bandit import BanditEnv

TimeStep = collections.namedtuple(
    "TimeStep", ["obs", "action_tm1", "expert_action_tm1", "reward", "done"]
)
EncLog = collections.namedtuple(
    "EncLog", ["loss", "recon_loss", "kl_loss", "latent_pred_loss", "grad_norm"]
)
PiLog = collections.namedtuple("PiLog", ["neglogp", "entropy", "grad_norm"])


class ReplayBuffer:
    def __init__(self, capacity, example_item):
        self.capacity = capacity
        self.storage = tree.map_structure(
            lambda x: np.zeros((capacity, *x.shape), dtype=x.dtype), example_item
        )
        self.idx = 0
        self.full = False
        self.episode_start_indices = set()

    def add_episode(self, episode):
        insert_index = (
            np.arange(self.idx, self.idx + episode.done.shape[0]) % self.capacity
        )

        def inserter(x, y):
            x[insert_index] = y

        self.episode_start_indices.difference_update(set(insert_index))
        self.episode_start_indices.add(self.idx)
        self.full = self.full or self.idx + episode.done.shape[0] >= self.capacity
        tree.map_structure(inserter, self.storage, episode)
        self.idx = (self.idx + episode.done.shape[0]) % self.capacity

    def sample(self, batch_size, traj_len, indices=None):
        if indices is None:
            start_indices = np.random.choice(list(self.episode_start_indices), batch_size)
        else:
            start_indices = np.array(list(indices))
        indices = (
            (start_indices[None] + np.arange(traj_len)[:, None]) % self.capacity
        ).reshape(-1)
        return tree.map_structure(
            lambda x: x[indices].reshape(traj_len, batch_size, *x.shape[1:]),
            self.storage,
        )


class ScrambleWrapper(Wrapper):
    def __init__(self, env, should_scramble=False):
        super().__init__(env)
        self.action_space = env.action_space
        self.should_scramble = should_scramble
        if self.should_scramble:
            self.permutation = np.array([[0, 1, 2, 3], [0, 2, 1, 3]])[
                self.env.np_random.randint(2)
            ]
        else:
            self.permutation = np.array([0, 1, 2, 3])
        self.inverse_permutation = np.argsort(self.permutation)
        self.observation_space = gym.spaces.Dict(
            {
                "obs": env.observation_space,
                "permutation": gym.spaces.MultiDiscrete((4, 4, 4, 4)),
                "inverse_permutation": gym.spaces.MultiDiscrete((4, 4, 4, 4)),
            }
        )

    def reset(self, **kwargs):
        if self.should_scramble:
            self.permutation = np.array([[0, 1, 2, 3], [0, 2, 1, 3]])[
                self.env.np_random.randint(2)
            ]
        else:
            self.permutation = np.array([0, 1, 2, 3])
        self.inverse_permutation = np.argsort(self.permutation)
        obs = self.env.reset(**kwargs)
        return {
            "obs": obs,
            "permutation": self.permutation,
            "inverse_permutation": self.inverse_permutation,
        }

    def step(self, action):
        action = self.permutation[action]
        o, r, d, i = self.env.step(action)
        i["permutation"] = self.permutation
        obs = {
            "obs": o,
            "permutation": self.permutation,
            "inverse_permutation": self.inverse_permutation,
        }
        return obs, r, d, i


class ActionNoiseWrapper(Wrapper):
    def __init__(self, env, uniform_action_probability=0.1):
        super().__init__(env)
        self.action_space = env.action_space
        self.uniform_action_probability = uniform_action_probability

    def step(self, action):
        action = (
            self.env.action_space.sample()
            if self.env.np_random.rand() < self.uniform_action_probability
            else action
        )
        o, r, d, i = self.env.step(action)
        return o, r, d, i


def pack_structure(xs, axis=0):
    return tree.map_structure(lambda *xs: np.stack(xs, axis=axis), *xs)


def get_expert_sample_fn(model, action_space):
    def sample_fn(obs, state=None, action_tm1=None, episode_start=None, deterministic=True, inject_noise_probability=0.0, frac=1.0):
        expert_action, state = model.predict(obs, state=state, deterministic=deterministic)
        if isinstance(action_space, gym.spaces.Discrete):
            random_action = np.random.randint(0, action_space.n, size=expert_action.shape, dtype=expert_action.dtype)
        elif isinstance(action_space, gym.spaces.Box):
            noise = np.random.normal(0, 0.1, size=expert_action.shape)
            random_action = np.clip(expert_action + noise, action_space.low, action_space.high, dtype=expert_action.dtype)
        choose = action_space.np_random.uniform(0, 1, size=expert_action.shape[0]) < (1 - inject_noise_probability)
        if choose.ndim == 1 and expert_action.shape[0] == choose.shape[0]:
            action = np.where(choose[:, np.newaxis], expert_action, random_action)
        else:
            action = np.where(choose, expert_action, random_action)
        return action, expert_action, state
    return sample_fn


def get_ortega_expert_sample_fn(action_space, expert_noise):
    def sample_fn(obs, state, action_tm1, episode_start, deterministic, inject_noise_probability, frac=1.0):
        best_actions = obs["best_arm"]
        random_actions = action_space.np_random.randint(0, action_space.n, size=best_actions.shape[0])
        choose_best = action_space.np_random.uniform(0, 1, size=best_actions.shape[0]) < (1-expert_noise) - expert_noise/4
        action = np.where(choose_best, best_actions, random_actions)
        return action, action, state

    return sample_fn


def get_bc_sample_fn(model, action_space, is_vectorized):
    def sample_fn(obs, state, action_tm1, episode_start, deterministic, inject_noise_probability, frac=1.0):
        with th.no_grad():
            action_tm1 = th.tensor(action_tm1, device=model.device).unsqueeze(0)
            obs_t = tree.map_structure(
                lambda x: th.tensor(x, device=model.device).unsqueeze(0), obs
            )
            distribution, state = model(state, obs_t, action_tm1)
            actions = distribution.get_actions(deterministic=deterministic)
            actions = actions.detach().cpu().numpy().reshape((-1,) + action_space.shape)
        if not is_vectorized:
            actions = actions[0]
        return actions, actions, state

    return sample_fn


def get_dagger_sample_fn(bc_sample_fn, expert_sample_fn):
    def sample_fn(obs, state, action_tm1, episode_start, deterministic, inject_noise_probability, frac=1.0):
        bc_action, _, state = bc_sample_fn(
            obs, state, action_tm1, episode_start, deterministic, inject_noise_probability=0.0,
        )
        # Ignores inject_noise_probability
        _, expert_action, _ = expert_sample_fn(
            obs, state, action_tm1, episode_start, deterministic, inject_noise_probability=0.0,
        )
        mask = np.random.uniform(0, 1, size=(action_tm1.shape[0],)) > frac
        action = np.select(mask, expert_action, bc_action)
        return action, expert_action, state

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
        ep_rew += reward

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
                rnn_state[:, i, :] = rnn_resetter(1)
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


def pack_structure(xs, axis=0):
    return tree.map_structure(lambda *xs: np.stack(xs, axis=axis), *xs)


def unpack_structure(structure, axis=0):
    transposed = tree.map_structure(lambda t: th.moveaxis(t, axis, 0), structure)
    flat = tree.flatten(transposed)
    unpacked = list(map(lambda xs: tree.unflatten_as(structure, xs), zip(*flat)))
    return unpacked


def encoder_losses(
    batch,
    device,
    bcmodel,
    n_pred_steps,
    kl_weight,
):
    # zero out everything after done
    batch_mask = np.concatenate(
        [np.ones_like(batch.done[:1]), np.cumprod(1 - batch.done, axis=0)], axis=0
    )[:-1]
    batch_mask = th.tensor(batch_mask, device=device)
    batch = tree.map_structure(lambda x: th.tensor(x, device=device), batch)
    T, B = batch.action_tm1.shape[0:2]
    rnn_state = bcmodel.init_hidden(B).to(device)
    action_tm1 = batch.action_tm1
    obs_t = batch.obs
    # encoder takes obs_t and action_tm1
    m, rnn_state = bcmodel.forward_encoder(
        rnn_state, tree.map_structure(lambda x: x[:-1], obs_t), action_tm1[:-1]
    )
    z = m.rsample()
    state_pred_losses = []
    predicted_latents = []
    input_obs = tree.map_structure(lambda x: x[:-1], obs_t)
    input_action = action_tm1[1:]
    input_z = z
    input_mask = batch_mask[1:]
    target_obs = tree.map_structure(lambda x: x[1:], obs_t)
    for _ in range(n_pred_steps):
        # decoder takes z_t, obs_t, and action_t
        pred_obs, pred_latent = bcmodel.forward_decoder(
            input_z, input_obs, input_action
        )
        for k in bcmodel.decoder_obs_keys:
            state_pred_loss = th.nn.functional.mse_loss(
                pred_obs[k].reshape(target_obs[k].shape).float(),
                target_obs[k].float(),
                reduction="none",
            )
            state_pred_loss = state_pred_loss.mean(
                dim=[i for i in range(2, len(state_pred_loss.shape))]
            )
            state_pred_loss = (state_pred_loss * input_mask).sum() / max(
                input_mask.sum(), 1.0
            )
            state_pred_losses.append(state_pred_loss)
        predicted_latents.append(pred_latent)
        input_obs = tree.map_structure(lambda x: x[1:], input_obs)
        input_action = input_action[1:]
        input_z = input_z[:-1]
        input_mask = input_mask[1:]
        target_obs = tree.map_structure(lambda x: x[1:], target_obs)
    if bcmodel.latent_pred_kwargs is not None:
        target_latent = batch.obs[bcmodel.latent_pred_kwargs["key"]][1:]
        if bcmodel.latent_pred_kwargs["loss"] == "mse":
            latent_pred_loss = (
                th.nn.functional.mse_loss(
                    predicted_latents[0].reshape(-1),
                    target_latent.reshape(-1).float(),
                    reduction="none",
                )
                .reshape(*predicted_latents[0].shape[:2], -1)
                .mean(dim=-1)
            )
        elif bcmodel.latent_pred_kwargs["loss"] == "cross_entropy":
            n_classes = bcmodel.latent_pred_kwargs["n_classes"]
            shape = predicted_latents[0].shape[:2]
            latent_pred_loss = (
                th.nn.functional.cross_entropy(
                    predicted_latents[0].reshape(-1, n_classes),
                    target_latent.reshape(-1).long(),
                    reduction="none",
                )
                .reshape(*shape[:2], -1)
                .mean(dim=-1)
            )
        else:
            raise NotImplementedError
        latent_pred_loss = (latent_pred_loss * batch_mask[1:]).sum() / max(
            batch_mask[1:].sum(), 1.0
        )
    else:
        latent_pred_loss = th.tensor(0.0, device=device)
    prev_mu = th.zeros((1, B, bcmodel.latent_dim)).to(device)
    prev_std = th.ones((1, B, bcmodel.latent_dim)).to(device)
    prior_mu = th.cat([prev_mu, m.mean[:-1]], dim=0)
    prior_std = th.cat([prev_std, m.stddev[:-1]], dim=0)
    prior = th.distributions.Normal(prior_mu, prior_std)
    kl_loss = (
        th.distributions.kl.kl_divergence(m, prior).sum(dim=-1) * batch_mask[1:]
    ).sum() / max(batch_mask[1:].sum(), 1.0)
    enc_loss = sum(state_pred_losses) + kl_loss * kl_weight + latent_pred_loss
    return enc_loss, state_pred_losses, kl_loss, latent_pred_loss


def encoder_update(
    imitation_env,
    bc_sample_fn,
    bcmodel,
    optimizer,
    buffer,
    n_train_steps,
    n_bptt_steps,
    n_pred_steps,
    stochastic,
    deterministic,
    n_episodes,
    max_episode_length,
    is_atari,
    n_envs,
    kl_weight,
    batch_size,
    grad_clip,
    schedule_timesteps,
    timesteps_so_far,
    should_train,
    logger,
):
    bcmodel.eval()
    bc_steps, bc_rewards, bc_lens, _ = sample_data(
        imitation_env,
        bc_sample_fn,
        bcmodel.init_hidden,
        True,
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
        buffer.add_episode(ep)
    if not should_train:
        return timesteps_so_far
    device = bcmodel.device
    logs = []
    bcmodel.train()
    norm_parameters = set(bcmodel.enc_parameters) - set(bcmodel.debug_decoder.parameters())
    for _ in range(n_train_steps):
        batch = buffer.sample(batch_size, n_bptt_steps)
        enc_loss, state_pred_losses, kl_loss, latent_pred_loss = encoder_losses(
            batch, device, bcmodel, n_pred_steps, kl_weight)
        optimizer.zero_grad()
        enc_loss.backward()
        if grad_clip is not None:
            th.nn.utils.clip_grad_norm_(norm_parameters, grad_clip)
        optimizer.step()
        total_norm = 0.0
        for p in norm_parameters:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm**2
        total_norm = total_norm**0.5
        enclog = tree.map_structure(
            lambda x: x.detach().cpu().numpy(),
            EncLog(enc_loss, sum(state_pred_losses), kl_loss, latent_pred_loss, total_norm),
        )
        logs.append(enclog)
    enclog = tree.map_structure(np.mean, pack_structure(logs))
    for k, v in enclog._asdict().items():
        logger.record(f"enc/{k}", v)
    logger.record("bc/rewards", np.mean(bc_rewards))
    logger.record("bc/lens", np.mean(bc_lens))
    return timesteps_so_far


def policy_losses(batch, device, bcmodel, deconfounding, use_hindsight):
    # zero out everything after done
    batch_mask = np.concatenate(
        [np.ones_like(batch.done[:1]), np.cumprod(1 - batch.done, axis=0)], axis=0
    )[:-1]
    batch_mask = th.tensor(batch_mask, device=device)
    batch = tree.map_structure(lambda x: th.tensor(x, device=device), batch)
    T, B = batch.action_tm1.shape[0:2]
    rnn_state = bcmodel.init_hidden(B).to(device)
    action_tm1 = batch.action_tm1
    expert_action_tm1 = batch.expert_action_tm1
    obs_t = batch.obs
    if deconfounding:
        # encoder takes obs_t and action_tm1
        m, rnn_state = bcmodel.forward_encoder(
            rnn_state, tree.map_structure(lambda x: x[:-1], obs_t), action_tm1[:-1]
        )
        z = m.mean.detach()
        if use_hindsight:
            fake_done = batch.done[:-1].float() * batch_mask[:-1].float()
            # If the episode didn't finish, use the last z
            fake_done[-1] = (~(th.sum(fake_done, dim=0) > 0)).float()
            z = th.ones_like(z) * (z * fake_done[..., None]).sum(
                axis=0, keepdims=True
            )
        action_dist, _ = bcmodel(
            rnn_state,
            tree.map_structure(lambda x: x[:-1], obs_t),
            action_tm1[:-1],
            z,
        )
    else:
        action_dist, _ = bcmodel(
            rnn_state, tree.map_structure(lambda x: x[:-1], obs_t), action_tm1[:-1]
        )
    flat_mask = batch_mask[:-1].reshape(-1)
    if hasattr(bcmodel.action_space, 'n'):
        logprob = action_dist.log_prob(expert_action_tm1[1:].reshape(-1))
    else:
        logprob = action_dist.log_prob(expert_action_tm1[1:].reshape((T-1)*B, -1))
    logprob = (logprob * flat_mask).sum() / max(flat_mask.sum(), 1.0)
    entropy = (action_dist.entropy() * flat_mask).sum() / max(flat_mask.sum(), 1.0)
    pi_loss = -logprob
    return pi_loss, entropy


def policy_update(
    expert_env,
    expert_sample_fn,
    bcmodel,
    bc_optimizer,
    buffer,
    n_train_steps,
    n_bptt_steps,
    batch_size,
    deconfounding,
    dagger,
    use_hindsight,
    stochastic,
    deterministic,
    n_episodes,
    max_episode_length,
    is_atari,
    n_envs,
    schedule_timesteps,
    timesteps_so_far,
    should_train,
    logger,
    inject_noise_probability,
):
    bcmodel.eval()
    device = bcmodel.device
    expert_steps, expert_rewards, expert_lens, timesteps_so_far = sample_data(
        expert_env,
        expert_sample_fn,
        bcmodel.init_hidden,
        stochastic=stochastic,
        deterministic=deterministic,
        n_episodes=n_episodes,
        max_episode_length=max_episode_length,
        progress=False,
        is_atari=is_atari,
        n_envs=n_envs,
        schedule_timesteps=schedule_timesteps,
        timesteps_so_far=timesteps_so_far,
        inject_noise_probability=inject_noise_probability,
    )
    for ep in expert_steps:
        buffer.add_episode(ep)
    if not should_train:
        return timesteps_so_far
    logs = []
    bcmodel.train()
    for i in range(n_train_steps):
        batch = buffer.sample(batch_size, n_bptt_steps)
        pi_loss, entropy = policy_losses(batch, device, bcmodel, deconfounding, use_hindsight)
        bc_optimizer.zero_grad()
        pi_loss.backward()
        bc_optimizer.step()
        total_norm = 0.0
        for p in bcmodel.pi_parameters:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm**2
        total_norm = total_norm**0.5
        pilog = tree.map_structure(
            lambda x: x.detach().cpu().numpy(), PiLog(pi_loss, entropy, total_norm)
        )
        logs.append(pilog)
    pilog = tree.map_structure(np.mean, pack_structure(logs))
    for k, v in pilog._asdict().items():
        logger.record(f"bc/{k}", v)
    logger.record("expert/rewards", np.mean(expert_rewards))
    logger.record("expert/lens", np.mean(expert_lens))
    return timesteps_so_far


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="ray or local", type=str, default="local")
    parser.add_argument("--env-name", help="environment ID", type=str, default="CartPole-v1")
    parser.add_argument("--imitation-env-name", help="environment ID", type=str)
    parser.add_argument("--imitator-feature-extractor-obs-keys", nargs="+", type=str, default=["obs"])
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
        "--deconfounding",
        action="store_true",
        default=False,
        help="Use deconfounding",
    )
    parser.add_argument(
        "--dagger",
        action="store_true",
        default=False,
        help="Use dagger",
    )
    parser.add_argument(
        "--schedule-timesteps",
        help="dagger schedule timesteps",
        default=1000000,
        type=int,
    )
    parser.add_argument(
        "--hindsight-policy-objective",
        action="store_true",
        default=False,
        help="Use hindsight for training policy",
    )
    parser.add_argument("--bptt-steps", help="bptt steps", default=50, type=int)
    parser.add_argument(
        "--enc-pred-steps", help="encoder prediction steps", default=1, type=int
    )
    parser.add_argument("--latent-dim", help="enc latent dim", default=32, type=int)
    parser.add_argument("--kl-weight", help="enc kl weight", default=1.0, type=float)
    parser.add_argument("--enc-lr", help="enc lr", default=1e-3, type=float)
    parser.add_argument("--enc-grad-clip", help="enc gradient clipping", type=float)
    parser.add_argument("--bc-lr", help="bc lr", default=1e-3, type=float)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--enc-train-steps", type=int, default=1)
    parser.add_argument("--bc-train-steps", type=int, default=1)
    parser.add_argument("--enc-batch-size", type=int, default=32)
    parser.add_argument("--bc-batch-size", type=int, default=32)
    parser.add_argument("--inject-noise-probability", help="injects noise to expert actions at this probability", default=0.0, type=float)
    parser.add_argument(
        "--buffer-size",
        help="how many trajectories to keep in memory",
        default=1000,
        type=int,
    )
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--traj-save-interval", type=int, default=0)
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    parser.add_argument("--logdir", help="Where to log", default="", type=str)
    args = parser.parse_args()
    return args


def save_trajs(trajs, path, bcmodel, rnn_resetter):
    device = bcmodel.device
    trajs = pack_structure(trajs, axis=1)
    T, B = trajs.done.shape[:2]
    with th.no_grad():
        rnn_states = rnn_resetter(B)
        action_tm1 = th.tensor(trajs.action_tm1, device=device)
        obs_t = tree.map_structure(
            lambda x: th.tensor(x, device=device), trajs.obs
        )
        distribution, _ = bcmodel(rnn_states, obs_t, action_tm1)
        logits = distribution.distribution.logits.reshape(T, B, -1).detach().cpu().numpy()
    trajs = dict(trajs._asdict())
    trajs['logits'] = logits
    with open(path, 'wb') as f:
        pickle.dump(trajs, f)


def enjoy(
    algo,
    bc_batch_size,
    bc_lr,
    bc_train_steps,
    bptt_steps,
    enc_pred_steps,
    buffer_size,
    deconfounding,
    dagger,
    schedule_timesteps,
    hindsight_policy_objective,
    deterministic,
    device,
    enc_batch_size,
    enc_lr,
    enc_grad_clip,
    enc_train_steps,
    env_name,
    imitation_env_name,
    epochs,
    exp_id,
    folder,
    kl_weight,
    latent_dim,
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
    imitator_feature_extractor_obs_keys,
    inject_noise_probability,
    *args,
    is_ray=False,
    **kwargs,
):
    logger = sb_logger.configure(os.path.join(logdir, "train"), ["stdout", "csv"])
    eval_logger = sb_logger.configure(os.path.join(logdir, "eval"), ["stdout", "csv"])

    env_name = EnvironmentName(env_name)
    algo = algo
    folder = folder

    if env_name.gym_id == "OrtegaBandit-v0":
        hyperparams = {}
        hyperparams["env_wrapper"] = ["sb3based.gail.IdentityWrapper"]
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
    feature_extractor_kwargs["obs_keys"] = imitator_feature_extractor_obs_keys
    if deconfounding:
        if env_name.gym_id == "LunarLander-v2":
            latent_pred_kwargs = {
                "key": "permutation",
                "pred_size": 16,
                "loss": "cross_entropy",
                "n_classes": 4,
            }
        elif env_name.gym_id == "HalfCheetahBulletEnv-v0":
            latent_pred_kwargs = {
                "key": "target",
                "pred_size": 1,
                "loss": "mse",
            }
        elif env_name.gym_id == "AntGoal-v0":
            latent_pred_kwargs = {
                "key": "target",
                "pred_size": 2,
                "loss": "mse",
            }
        elif env_name.gym_id == "OrtegaBandit-v0":
            latent_pred_kwargs = {
                "key": "best_arm",
                "pred_size": 5,
                "loss": "cross_entropy",
                "n_classes": 5,
            }
        else:
            latent_pred_kwargs = None
        bcmodel = DeconfoundedRNNPolicy(
            imitation_env.observation_space,
            imitation_env.action_space,
            lr_schedule=lambda x: 0,
            latent_dim=latent_dim,
            net_arch=None,
            features_extractor_class=feature_extractor_class,
            features_extractor_kwargs=feature_extractor_kwargs,
            latent_pred_kwargs=latent_pred_kwargs,
        )
        bc_optimizer = th.optim.AdamW(bcmodel.pi_parameters, lr=bc_lr)
        enc_optimizer = th.optim.AdamW(bcmodel.enc_parameters, lr=enc_lr)
    else:
        bcmodel = RNNPolicy(
            imitation_env.observation_space,
            imitation_env.action_space,
            lr_schedule=lambda x: 0,
            net_arch=None,
            features_extractor_class=feature_extractor_class,
            features_extractor_kwargs=feature_extractor_kwargs,
        )
        bc_optimizer = th.optim.AdamW(bcmodel.parameters(), lr=bc_lr)
        enc_optimizer = None

    bcmodel.to(device)
    example_timestep = TimeStep(
        tree.map_structure(lambda t: t[0], imitation_env.reset()),
        np.zeros_like(imitation_env.action_space.sample()),
        np.zeros_like(imitation_env.action_space.sample()),
        np.zeros(1)[0],
        np.zeros(1)[0],
    )
    if deconfounding:
        encoder_buffer = ReplayBuffer(buffer_size, example_timestep)
    policy_buffer = ReplayBuffer(buffer_size, example_timestep)
    bc_sample_fn = get_bc_sample_fn(bcmodel, imitation_env.action_space, True)
    timesteps_so_far = 0
    if dagger:
        expert_sample_fn = get_dagger_sample_fn(bc_sample_fn, expert_sample_fn)
    for i in range(epochs):
        should_train = i >= warmup_epochs
        if should_train:
            bcmodel.eval()
            trajs, rewards, lens, _ = sample_data(
                imitation_env,
                bc_sample_fn,
                bcmodel.init_hidden,
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
            dummy_buffer = ReplayBuffer(n_episodes * (max_episode_length + 1), example_timestep)
            for ep in trajs:
                dummy_buffer.add_episode(ep)
            r = np.mean(rewards)
            l = np.mean(lens)
            eval_logger.record("bc/rewards", r)
            eval_logger.record("bc/lens", l)
            if deconfounding:
                batch = dummy_buffer.sample(n_episodes, max_episode_length, dummy_buffer.episode_start_indices)
                enc_loss, state_pred_losses, kl_loss, latent_pred_loss = encoder_losses(
                    batch, device, bcmodel, enc_pred_steps, kl_weight)
                enclog = tree.map_structure(
                    lambda x: x.detach().cpu().numpy(),
                    EncLog(enc_loss, sum(state_pred_losses), kl_loss, latent_pred_loss, th.tensor(0.0)),
                )
                for k, v in enclog._asdict().items():
                    eval_logger.record(f"enc/{k}", v)
            expert_trajs, rewards, lens, _ = sample_data(
                imitation_env,
                expert_sample_fn,
                bcmodel.init_hidden,
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
            dummy_buffer = ReplayBuffer(n_episodes * (max_episode_length + 1), example_timestep)
            for ep in trajs:
                dummy_buffer.add_episode(ep)
            batch = dummy_buffer.sample(n_episodes, max_episode_length, dummy_buffer.episode_start_indices)
            pi_loss, entropy = policy_losses(
                batch, device, bcmodel, deconfounding,
                hindsight_policy_objective)
            eval_logger.record("bc/neglogp", pi_loss.detach().cpu().numpy())
            eval_logger.record("bc/entropy", entropy.detach().cpu().numpy())
            eval_logger.dump()
            if is_ray:
                session.report({"rewards": r, "lens": l})
            if i % 5000 == 0 or i == epochs - 1:
                bcmodel.save(os.path.join(log_dir, f"bcmodel_{i}"))
            if traj_save_interval > 0 and (i % traj_save_interval == 0 or i == epochs - 1):
                save_trajs(trajs, os.path.join(log_dir, f"trajs_{i}.pkl"), bcmodel, bcmodel.init_hidden)
                save_trajs(expert_trajs, os.path.join(log_dir, f"expert_trajs_{i}.pkl"), bcmodel, bcmodel.init_hidden)
        if deconfounding:
            _ = encoder_update(
                imitation_env=imitation_env,
                bc_sample_fn=bc_sample_fn,
                bcmodel=bcmodel,
                optimizer=enc_optimizer,
                buffer=encoder_buffer,
                n_train_steps=enc_train_steps,
                n_bptt_steps=bptt_steps,
                n_pred_steps=enc_pred_steps,
                batch_size=enc_batch_size,
                stochastic=stochastic,
                deterministic=deterministic,
                n_episodes=n_episodes,
                max_episode_length=max_episode_length,
                is_atari=is_atari,
                n_envs=n_envs,
                kl_weight=kl_weight,
                grad_clip=enc_grad_clip,
                schedule_timesteps=schedule_timesteps,
                timesteps_so_far=timesteps_so_far,
                should_train=should_train,
                logger=logger,
            )
        new_timesteps = policy_update(
            expert_env=imitation_env,
            expert_sample_fn=expert_sample_fn,
            bcmodel=bcmodel,
            bc_optimizer=bc_optimizer,
            buffer=policy_buffer,
            n_train_steps=bc_train_steps,
            n_bptt_steps=bptt_steps,
            batch_size=bc_batch_size,
            deconfounding=deconfounding,
            dagger=dagger,
            use_hindsight=hindsight_policy_objective,
            stochastic=stochastic,
            deterministic=deterministic,
            n_episodes=n_episodes,
            max_episode_length=max_episode_length,
            is_atari=is_atari,
            n_envs=n_envs,
            schedule_timesteps=schedule_timesteps,
            timesteps_so_far=timesteps_so_far,
            should_train=should_train,
            logger=logger,
            inject_noise_probability=inject_noise_probability,
        )
        timesteps_so_far = new_timesteps
        logger.record("train/step", i)
        logger.record("train/expert_timesteps", timesteps_so_far)
        logger.dump()

    imitation_env.close()
    return bcmodel



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
