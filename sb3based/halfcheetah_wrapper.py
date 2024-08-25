import gym
import numpy as np

import pybullet_envs

class HCWrapper(gym.Wrapper):
    def __init__(self, env, target_vis=False, indicator_noise=0.2):
        super().__init__(env)
        self.target = np.random.uniform(0.0, 3.0)
        self.env = env
        self.env.reset()
        self.target_vis = target_vis
        self.indicator_noise = indicator_noise
        self.dt = self.env.env.robot.scene.dt
        self.observation_space = gym.spaces.Dict(
            {
                "obs": env.observation_space,
                "indicator": gym.spaces.Box(low=-1.0, high=1.0, shape=(1,)),
                "target": gym.spaces.Box(low=-1.0, high=1.0, shape=(1,))
            }
        )

    def reset(self):
        self.target = np.random.uniform(0.0, 3.0)
        obs = self.env.reset()
        return {"obs": obs, "indicator": np.array([-1.0]), "target": np.array([self.target])}

    def step(self, action):
        p = self.env.robot.robot_body.get_pose()[0]
        next_obs, _, done, info = self.env.step(action)
        p_prime = self.env.robot.robot_body.get_pose()[0]
        vel = (p_prime - p) / (self.dt)
        ind = 1 if vel > self.target else -1.0
        ind = ind if self.env.np_random.rand() < self.indicator_noise else -1.0 * ind
        next_obs = {"obs": next_obs, "indicator": np.array([ind]), "target": np.array([self.target])}
        forward_reward = -1.0 * abs(vel - self.target)
        ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(action))
        reward = forward_reward - ctrl_cost
        return next_obs, reward, done, info
