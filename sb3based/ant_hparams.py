import torch

from sb3based.rnnpolicy import KeyCombinedExtractor

hyperparams = {
    "AntBulletEnv-v0": {
        "env_wrapper": [
            {"sb3based.ant_wrapper.AntWrapper": {"target_vis": True}},
        ],
        "policy": "MultiInputPolicy",
        "vf_coef": 0.5,
        "n_envs": 16,
        "n_timesteps": 3e6,
        "n_steps": 512,
        "batch_size": 128,
        "gae_lambda": 0.9,
        "gamma": 0.99,
        "n_epochs": 20,
        "ent_coef": 0.0,
        "sde_sample_freq": 4,
        "max_grad_norm": 0.5,
        "use_sde": True,
        "clip_range": 0.4,
        "learning_rate": 3e-5,
        "policy_kwargs": dict(
            log_std_init=-1,
            ortho_init=False,
            activation_fn=torch.nn.ReLU,
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            features_extractor_class=KeyCombinedExtractor,
            features_extractor_kwargs=dict(obs_keys=["obs", "indicator", "target"]),
        ),
        "normalize": {"norm_obs_keys":["obs"]},
    },
    "Ant-v3": {
        "env_wrapper": [
            {"sb3based.ant_wrapper.MujocoAntWrapper": {"target_vis": True}},
        ],
        "policy": "MultiInputPolicy",
        "vf_coef": 0.5,
        "n_envs": 16,
        "n_timesteps": 3e6,
        "n_steps": 512,
        "batch_size": 128,
        "gae_lambda": 0.9,
        "gamma": 0.99,
        "n_epochs": 20,
        "ent_coef": 0.0,
        "sde_sample_freq": 4,
        "max_grad_norm": 0.5,
        "use_sde": True,
        "clip_range": 0.4,
        "learning_rate": 3e-5,
        "policy_kwargs": dict(
            log_std_init=-1,
            ortho_init=False,
            activation_fn=torch.nn.ReLU,
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            features_extractor_class=KeyCombinedExtractor,
            features_extractor_kwargs=dict(obs_keys=["obs", "indicator", "target"]),
        ),
        "normalize": {"norm_obs_keys":["obs"]},
    },
    "AntNoTerminate-v3": {
        "normalize": True,
        "n_timesteps": 3e6,
        "policy": "MlpPolicy",
    }
}
