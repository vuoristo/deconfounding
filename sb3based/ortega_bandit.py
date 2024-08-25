from typing import Optional, List

import numpy as np

from gym import Env, spaces


class BanditEnv(Env):
    """
    This environment is a POMDP with aliased states. The environment
    corresponds to a 5-armed bandit where the latent variable decides
    the best arm. Each of the five arms has equal chance to be the
    best arm.

    Observation is 1 with probability 0.75 if the best arm was
    selected, 1 with probability 0.25 for the other arms, and 0
    otherwise.
    """

    def __init__(self, num_arms=5, p_true_obs=0.75, latent=None, context=None):
        self.num_arms = num_arms
        self.p_true_obs = p_true_obs
        self.observation_space = spaces.Dict(
            {
                "obs": spaces.Box(0, 1, shape=(1,), dtype=np.float32),
                "best_arm": spaces.Discrete(5)
            }
        )
        self.action_space = spaces.Discrete(num_arms)
        if latent is not None:
            self.fixed_latent = True
            self.latent = latent
        else:
            self.fixed_latent = False
            self.latent = 0

    def step(self, a):
        r = 0.0
        d = False
        if a == self.latent:
            o = (np.random.uniform() < self.p_true_obs) * 1.0
            r = 1.0
        else:
            o = (np.random.uniform() > self.p_true_obs) * 1.0
            r = 0.0
        obs = {"obs": np.array([o]), "best_arm": self.latent}
        return (obs, r, d, {"latent": self.latent})

    def reset(
        self,
        *,
        # seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
        latent: Optional[np.ndarray] = None,
    ):
        if self.fixed_latent:
            return self.reset_with_latent(self.latent)
        return self.reset_with_latent(latent)

    def reset_with_latent(
        self,
        latent,
    ):
        if latent is None:
            latent = np.random.randint(0, self.num_arms)
        self.latent = latent
        obs = {"obs": np.zeros((1,)), "best_arm": self.latent}
        return obs

    def set_should_reset_latent(self, should_reset_latent):
        self.should_reset_latent = should_reset_latent


if __name__ == "__main__":
    env = BanditEnv()
    obs = env.reset()
    print(obs)
    while True:
        action = int(input())
        if action not in [0, 1, 2, 3, 4]:
            break
        obs, r, d, i = env.step(action)
        print(obs)
        print(f"reward {r}")
        if d == True:
            obs = env.reset()
            print(obs)
