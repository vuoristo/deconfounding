from sb3based.rnnpolicy import KeyCombinedExtractor

hyperparams = {
    "LunarLanderFixedLength-v2": {
        "env_wrapper": [
            {"sb3based.rnnbc.ActionNoiseWrapper": {"uniform_action_probability": 0.1}},
            {"sb3based.rnnbc.ScrambleWrapper": {"should_scramble": True}},
            "sb3based.gail.ObsActionWrapper",
        ],
        "policy": "MultiInputPolicy",
        "policy_kwargs": {
            "features_extractor_class": KeyCombinedExtractor,
            "features_extractor_kwargs": {"obs_keys": ["obs", "permutation"]},
        },
        "vf_coef": 0.5,
        "n_envs": 16,
        "n_timesteps": 2e6,
        "n_steps": 1000,
        "batch_size": 64,
        "gae_lambda": 0.98,
        "gamma": 0.999,
        "n_epochs": 4,
        "ent_coef": 0.01,
        "normalize": {"norm_obs_keys":["obs"]},
    }
}

hyperparams["LunarLander-v2"] = hyperparams["LunarLanderFixedLength-v2"]