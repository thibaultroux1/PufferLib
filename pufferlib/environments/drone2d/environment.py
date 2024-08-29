import gymnasium

import functools

import pufferlib.emulation
import pufferlib.postprocess


ALIASES = {
    'drone2d': 'Drone2D-discrete-v0',
}

def env_creator(name='minigrid'):
    return functools.partial(make, name=name)

def make(name, reward_crash= -10, render_mode=None):
    if name in ALIASES:
        name = ALIASES[name]
    _ = pufferlib.environments.try_import('phd.env')
    if render_mode == "human":
        env = gymnasium.make(name, reward_crash=reward_crash, render_sim=True)
    else:
        env = gymnasium.make(name, reward_crash=reward_crash)
    env = pufferlib.postprocess.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env)
