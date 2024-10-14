from pdb import set_trace as T
from pettingzoo.utils.conversions import aec_to_parallel_wrapper
import functools

import pufferlib.emulation
import pufferlib.environments
import pufferlib.wrappers


def env_creator(name):
    return functools.partial(make, name)

def make(name, num_aircraft=15):
    from phd.env.MA_Airplane.ma_environment import MultiAgentEnvironment

    print(num_aircraft)
    env = MultiAgentEnvironment(width=400, height=400, num_aircraft=num_aircraft)
    return pufferlib.emulation.PettingZooPufferEnv(env=env)
