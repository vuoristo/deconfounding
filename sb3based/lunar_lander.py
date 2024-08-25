
from gym.envs.box2d.lunar_lander import LunarLander

class LunarLanderFixedLen(LunarLander):
    def __init__(self, max_episode_length=500):
        super().__init__()
        self.num_steps = 0
        self.has_terminated = False
        self.max_episode_length = max_episode_length

    def step(self, action):
        self.num_steps += 1
        o,r,d,i = super().step(action)
        r = 0 if self.has_terminated else r
        self.has_terminated = self.has_terminated or d
        d = self.num_steps > self.max_episode_length
        return o,r,d,i

    def reset(self):
        self.num_steps = 0
        self.has_terminated = False
        return super().reset()
