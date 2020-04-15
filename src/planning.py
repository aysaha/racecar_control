import os

import numpy as np
from scipy import optimize

from learning import dynamics

FILE = os.path.basename(__file__)
DIRECTORY = os.path.dirname(__file__)

def plan_trajectory(model, env):
    print("[{}] planning trajectory ({})".format(FILE))

    track = np.array(env.track)
    q_min = np.array(env.observation_space.low)
    q_max = np.array(env.observation_space.high)
    u_min = np.array(env.action_space.low)
    u_max = np.array(env.action_space.high)

    # TODO
    raise NotImplementedError