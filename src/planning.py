import os

import numpy as np

FILE = os.path.basename(__file__)
DIRECTORY = os.path.dirname(__file__)

def plan_trajectory(env, T=15):
    #print("[{}] planning trajectory".format(FILE))

    waypoints = np.array(env.track)
    M = waypoints.shape[0]
    N = int(T/env.dt)
    trajectory = np.zeros((N, 3))

    for i in range(N):
        j = int(i * M/N)
        alpha = (i * M/N) - j
        theta, x, y = (1-alpha)*waypoints[j, 1:] + alpha*waypoints[(j+1) % M, 1:]
        theta = (theta + np.pi/2) % (2*np.pi)
        trajectory[i] = [x, y, theta]

    return trajectory
