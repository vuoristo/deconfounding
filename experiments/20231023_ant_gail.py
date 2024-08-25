import os

from ray import tune
from ray.air import RunConfig

from sb3based.gail import trainable

trial_space = {
    "algo": "ppo",
    "buffer_size": 1000000,
    "deterministic": True,
    "device": "cuda",
    "disc_grad_clip": 100.0,
    "disc_lr": tune.grid_search([1e-5, 1e-4,]),
    "disc_train_steps": tune.grid_search([20]),
    "env_name": "AntGoal-v0",
    "epochs": 300,
    "exp_id": 16,
    "folder": os.path.join(os.getcwd(), "experts"),
    "imitation_env_name": "AntGoal-v0",
    "imitator_feature_extractor_obs_keys": ["obs", "action_tm1", "indicator"],
    "imitator_gamma": 0.99,
    "imitator_gae_lambda": 0.95,
    "imitator_learning_rate": tune.grid_search([3e-4,]),
    "imitator_batch_size": 400,
    "imitator_n_epochs": 10,
    "imitator_max_grad_norm": 0.5,
    "imitator_train_steps": 1,
    "imitator_normalize_advantage": True,
    "reward_net_inputs": tune.grid_search([["next_state"]]),
    "reward_net_obs_keys": ["obs", "indicator"],
    "load_best": True,
    "load_checkpoint": None,
    "load_last_checkpoint": False,
    "max_episode_length": 200,
    "mode": "ray",
    "n_envs": 64,
    "n_episodes": 64,
    "norm_reward": False,
    "num_threads": -1,
    "schedule_timesteps": 10000,
    "seed": tune.grid_search([
        2342348, 4127479, 13355,
        423426, 99955, 123123,
    ]),
    "stochastic": False,
    "warmup_epochs": 20,
    "traj_save_interval": 0,
    "inject_noise_probability": 0.0,
}

trainable = tune.with_resources(trainable, {"gpu": 0.5})
tuner = tune.Tuner(
    trainable,
    param_space=trial_space,
    run_config=RunConfig(local_dir="/data/antlia/risrio/dil/logs/20231023_ant_gail_6_seeds"),
)
tuner.fit()
