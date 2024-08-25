import os

from ray import tune
from ray.air import RunConfig

from sb3based.rnnbc import trainable

trial_space = {
    "algo": "ppo",
    "bc_batch_size": 16,
    "bc_lr": 3e-4,
    "bc_train_steps": 100,
    "bptt_steps": 500,
    "buffer_size": 2000000,
    "dagger": False,
    "deconfounding": True,
    "deterministic": True,
    "device": "cuda",
    "enc_batch_size": 16,
    "enc_grad_clip": 100.0,
    "enc_lr": 3e-4,
    "enc_pred_steps": 50,
    "enc_train_steps": 100,
    "env_name": "LunarLander-v2",
    "epochs": 300,
    "exp_id": 1,
    "folder": os.path.join(os.getcwd(), "experts"),
    "hindsight_policy_objective": False,
    "imitation_env_name": "LunarLanderFixedLength-v2",
    "imitator_feature_extractor_obs_keys": ["obs"],
    "kl_weight": 1.0,
    "latent_dim": 64,
    "load_best": True,
    "load_checkpoint": None,
    "load_last_checkpoint": False,
    "max_episode_length": 500,
    "mode": "ray",
    "n_envs": 16,
    "n_episodes": 16,
    "norm_reward": False,
    "num_threads": -1,
    "schedule_timesteps": 1000000,
    "seed": tune.grid_search([2342348, 4127479, 13355, 423426, 99955]),
    "stochastic": False,
    "warmup_epochs": 20,
}

trainable = tune.with_resources(trainable, {"gpu": 1.0})
tuner = tune.Tuner(trainable, param_space=trial_space, run_config=RunConfig(local_dir="logs/20230512_lunar_lander_dil"))
tuner.fit()

