import collections
import random

import numpy as np
import gym
import math
from gym.envs.mujoco import AntEnv


class AntGoalEnv(AntEnv):
    """
    Similar to
    https://github.com/lmzintgraf/varibad/blob/master/environments/mujoco/ant_goal.py
    but with a 10/3 times larger goal range, no control cost, and observation changes
    """
    def __init__(self, max_episode_steps=200, indicator_noise=0.1):
        self.indicator_noise = indicator_noise
        self._max_episode_steps = max_episode_steps
        self.task_dim = 2
        self.frame_stats = collections.defaultdict(float)
        self.seed()
        self.reset_task(None)
        super(AntGoalEnv, self).__init__()

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        xposafter = np.array(self.get_body_com("torso"))

        goal_reward = -np.sum(np.abs(xposafter[:2] - self.target))  # make it happy, not suicidal

        # No control cost because otherwise the ant doesn't like to go anywhere.
        ctrl_cost = .0 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 0.0
        reward = goal_reward - ctrl_cost - contact_cost + survive_reward
        done = False
        ob = self._get_obs()

        dx = self.target[0] - xposafter[0]
        dy = self.target[1] - xposafter[1]
        distance = (dx**2 + dy**2)**0.5
        heading = math.atan2(dy, dx)
        indicator = np.array([distance, heading]) + self.np_random.normal(size=2) * self.indicator_noise
        observation = {"obs": ob, "indicator": np.array([indicator]), "target": np.array([self.target])}
        self.frame_stats.update({
            "goal_reward": goal_reward,
            "indicator": indicator,
            "target": self.target,
            "contact_cost": contact_cost,
            "ctrl_cost": ctrl_cost,
            "reward": reward,
        })
        return observation, reward, done, dict(
            goal_forward=goal_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            task=self.get_task()
        )

    def reset(self):
        self.reset_task(None)
        obs = super().reset()
        return {"obs": obs, "indicator": np.array([[1.0, 1.0]]), "target": np.array([self.target])}

    def sample_tasks(self, num_tasks):
        a = np.array([random.random() for _ in range(num_tasks)]) * 2.0 * np.pi
        r = 10 * np.array([random.random() for _ in range(num_tasks)]) ** 0.5
        return np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)

    def set_task(self, task):
        self.target = task

    def get_task(self):
        return np.array(self.target)

    def reset_task(self, task):
        if task is None:
            task = self.sample_tasks(1)[0]
        self.set_task(task)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    def render(self, render_mode):
        # Import here so that this doesn't get loaded if we are not rendering
        import cv2
        frame = super().render(render_mode)
        if render_mode == 'human':
            return
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        # required for the cv2 functions to work correctly.
        frame = np.ascontiguousarray(frame)
        text_color = (0, 0, 0)
        y_offset = 20
        for k, v in self.frame_stats.items():
            cv2.putText(frame, f"{k}: {v}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
            y_offset += 20
        return frame
