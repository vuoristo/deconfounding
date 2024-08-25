import os

from ray import tune
from ray.air import RunConfig

from sb3based.rnnbc import trainable

trial_space = {
    "algo": "ppo",
    "bc_batch_size": 32,
    "bc_lr": 3e-4,
    "bc_train_steps": 100,
    "bptt_steps": 101,
    "buffer_size": 2000000,
    "dagger": False,
    "deconfounding": True,
    "deterministic": False,
    "device": "cuda",
    "enc_batch_size": 32,
    "enc_grad_clip": 100.0,
    "enc_lr": 3e-4,
    "enc_pred_steps": 1,
    "enc_train_steps": 1,
    "env_name": "OrtegaBandit-v0",
    "epochs": 2000,
    "exp_id": 1,
    "folder": os.path.join(os.getcwd(), "experts"),
    "hindsight_policy_objective": False,
    "imitation_env_name": None,
    "imitator_feature_extractor_obs_keys": ["obs"],
    "kl_weight": tune.grid_search([0.01]),
    "latent_dim": 5,
    "load_best": True,
    "load_checkpoint": None,
    "load_last_checkpoint": False,
    "max_episode_length": 100,
    "mode": "ray",
    "n_envs": 16,
    "n_episodes": 16,
    "norm_reward": False,
    "num_threads": -1,
    "schedule_timesteps": 1000000,
    "seed": tune.grid_search([2342348, 4127479, 13355,
                              423426, 99955, 34567, 5678765, 7899234, 629338, 8565959
                              ]),
    "stochastic": True,
    "warmup_epochs": 20,
    "traj_save_interval": 500,
}

trainable = tune.with_resources(trainable, {"gpu": 0.5})
tuner = tune.Tuner(
    trainable,
    param_space=trial_space,
    run_config=RunConfig(
        local_dir="logs/20230724_dil_bandit_enc_train_steps_1_mse_loss",
        ))
tuner.fit()

