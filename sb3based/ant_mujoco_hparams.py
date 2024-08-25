import torch
from sb3based.rnnpolicy import KeyCombinedExtractor


hyperparams = {
    "AntGoal-v0": {
        "env_wrapper": [],
        "policy": "MultiInputPolicy",
        "n_envs": 16,
        "n_timesteps": 5e6,
        "n_steps": 500,
        "learning_rate": 2.0633e-05,
        "policy_kwargs": dict(
            log_std_init=-2,
            ortho_init=False,
            activation_fn=torch.nn.ReLU,
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            features_extractor_class=KeyCombinedExtractor,
            features_extractor_kwargs=dict(obs_keys=["obs", "target"]),
        ),
        "normalize": {"norm_obs_keys":["obs", "indicator", "target"]},
    }
}
