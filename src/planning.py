import os

import numpy as np

FILE = os.path.basename(__file__)
DIRECTORY = os.path.dirname(__file__)

def plan_trajectory(env, T=20):
    print("[{}] planning trajectory".format(FILE))

    # parameterize trajectory with respect to time
    waypoints = np.array(env.track)
    M = waypoints.shape[0]
    N = int(T/env.dt)
    trajectory = np.zeros((N, 6))

    # target positions
    for i in range(N):
        j = int(i * M/N)
        alpha = (i * M/N) - j
        theta, x, y = (1-alpha)*waypoints[j, 1:] + alpha*waypoints[(j+1) % M, 1:]
        theta = (theta + np.pi/2) % (2*np.pi)
        trajectory[i, :3] = [x, y, theta]

    # target velocities
    for i in range(N):
        v_x, v_y, omega = (trajectory[(i+1) % N, :3] - trajectory[i, :3]) / env.dt
        trajectory[i, 3:] = [v_x, v_y, omega]

    return trajectory
