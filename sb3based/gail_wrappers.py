import gym
from gym import Wrapper
import numpy as np

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
