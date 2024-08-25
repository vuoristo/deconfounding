import gym
import numpy as np

# Make sure the environment thread has to import this before making the environment
import pybullet_envs


class AntWrapper(gym.Wrapper):
    def __init__(self, env, target_vis=False):
        super().__init__(env)
        self.target = np.random.uniform(0.0, 1.5)
        self.env = env
        self.env.reset()
        self.target_vis = target_vis
        self.dt = self.env.env.robot.scene.dt
        self.observation_space = gym.spaces.Dict(
            {
                "obs": env.observation_space,
                "indicator": gym.spaces.Box(low=-1.0, high=1.0, shape=(1,)),
                "target": gym.spaces.Box(low=-1.0, high=1.0, shape=(1,))
            }
        )
        self.t = 0

    def reset(self):
        self.target = np.random.uniform(0.1, 1.5)
        obs = self.env.reset()
        self.t = 0
        return {"obs": obs, "indicator": np.array([-1.0]), "target": np.array([self.target])}

    def step(self, action):
        self.t += 1
        p = self.env.env.parts['torso'].get_position()[0]
        next_obs, _, done, info = self.env.step(action)
        p_prime = self.env.env.parts['torso'].get_position()[0]
        vel = (p_prime - p) / self.dt
#         print(vel, self.target)
        ind = self.target if self.t > 200 else 0.0
        next_obs = {"obs": next_obs, "indicator": np.array([ind]), "target": np.array([self.target])}
        forward_reward = -1.0 * abs(vel - self.target)
        ctrl_cost = 0.1 * np.sum(np.square(action))
        living_reward = 1.75 # extra +0.75 to cancel the target
        reward = living_reward + forward_reward - ctrl_cost
        return next_obs, reward, done, info


class MujocoAntWrapper(gym.Wrapper):
    def __init__(self, env, target_vis=False):
        super().__init__(env)
        self.target = np.random.uniform(0.0, 3.0)
        self.env = env
        self.env.reset()
        self.target_vis = target_vis
        self.observation_space = gym.spaces.Dict(
            {
                "obs": env.observation_space,
                "indicator": gym.spaces.Box(low=-1.0, high=1.0, shape=(1,)),
                "target": gym.spaces.Box(low=-1.0, high=1.0, shape=(1,))
            }
        )
        self.t = 0

    def reset(self):
        self.target = np.random.uniform(0.1, 3.0)
        obs = self.env.reset()
        self.t = 0
        return {"obs": obs, "indicator": np.array([-1.0]), "target": np.array([self.target])}

    def step(self, action):
        self.t += 1
        xy_position_before = self.env.get_body_com("torso")[:2].copy()
        next_obs, reward, done, info = self.env.step(action)
        xy_position_after = self.env.get_body_com("torso")[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.env.dt
        x_velocity, y_velocity = xy_velocity

        ctrl_cost = self.env.control_cost(action)
        contact_cost = self.env.contact_cost

        forward_reward = x_velocity
        # forward_reward = -1.0 * abs(x_velocity - self.target) + 0.7
        # forward_reward = -1.0 * (xy_velocity)
        # forward_reward = -1.0 * (np.linalg.norm(xy_velocity) - self.target) ** 2 + 1.5
        healthy_reward = self.env.healthy_reward

        rewards = forward_reward + healthy_reward
        costs = ctrl_cost + contact_cost

        reward = rewards - costs
        print(f"xyv: {np.linalg.norm(xy_velocity):5.2f} tgt: {self.target:5.2f} fr: {forward_reward:5.2f} hr: {healthy_reward:5.2f} ctrl: {ctrl_cost:5.2f} ctct: {contact_cost:5.2f} reward {reward:5.2f}")

        ind = self.target if self.t > 200 else 0.0
        next_obs = {"obs": next_obs, "indicator": np.array([ind]), "target": np.array([self.target])}
        return next_obs, reward, done, info