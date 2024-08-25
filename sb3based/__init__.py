from gym.envs.registration import (
    registry,
    register,
    make,
    spec,
    load_env_plugins as _load_env_plugins,
)


register(
    id="LunarLanderFixedLength-v2",
    entry_point="sb3based.lunar_lander:LunarLanderFixedLen",
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id='OrtegaBandit-v0',
    entry_point="sb3based.ortega_bandit:BanditEnv",
    max_episode_steps=100,
)


register(
    id="AntNoTerminate-v3",
    entry_point="gym.envs.mujoco.ant_v3:AntEnv",
    max_episode_steps=200,
    reward_threshold=6000.0,
    kwargs=dict(terminate_when_unhealthy=False),
)

register(
    'AntGoal-v0',
    entry_point='sb3based.ant_mujoco:AntGoalEnv',
    max_episode_steps=200
)
