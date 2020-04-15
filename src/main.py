#!/usr/bin/env python

import os
import argparse

import gym
import numpy as np
import matplotlib.pyplot as plt

from controllers import initialize_controller
from learning import horizon, save_dataset, load_dataset, load_model
from utils import *

FILE = os.path.basename(__file__)
DIRECTORY = os.path.dirname(__file__)

TRACK_COLOR = '#666666'
CAR_COLOR = '#CC6600'
MODEL_COLOR = '#0066CC'
MAJOR_GRID_COLOR = '#CCCCCC'
MINOR_GRID_COLOR = '#DDDDDD'

def plot_trajectory(track, car, model=None, data=None, N=1):
    print("[{}] plotting trajectory".format(FILE))
    plt.figure(num='state_trajectory')

    # create plot title
    title = 'State Trajectory'

    # plot track
    track = np.vstack((track, track[0]))
    plt.plot(track[:, 2], track[:, 3], '-', linewidth=16, color=TRACK_COLOR, label="Track")

    # plot car
    car = np.array(car)
    plt.plot(car[:, 0], car[:, 1], '-', linewidth=2, color=CAR_COLOR, label="Car")

    # plot model
    if model is not None and data is not None:
        title += ' (N = {})'.format(N)
        q, u = np.array(data[0]), np.array(data[1])
        init, pred = [], []

        # run model over prediction horizon
        for i in range(q.shape[0] // N):
            if N > 1:
                init.append(q[N*i])
                pred.append(q[N*i])
            
            pred.extend(list(horizon(q[N*i], u[N*i:N*(i+1)], model, N)))

        # plot initial values
        if N > 1:
            init = np.array(init)
            plt.plot(init[:, 0], init[:, 1], 'o', markersize=4, color=MODEL_COLOR)
        
        # plot predicted values
        pred = np.array(pred)
        plt.plot(pred[:, 0], pred[:, 1], '--', linewidth=2, color=MODEL_COLOR, label="Model")

    # format labels
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(borderpad=1.2, handletextpad=1.2)

    # format grid
    plt.grid(which='major', color=MAJOR_GRID_COLOR)
    plt.grid(which='minor', color=MINOR_GRID_COLOR)
    plt.minorticks_on()

    plt.show()

def main(args):
    assert args.dataset is None or os.path.splitext(args.dataset)[1] == '.npz'
    assert args.model in ['nonlinear', 'affine', 'linear']
    assert args.control in ['robot', 'keyboard', 'xbox']

    # create environment
    print("[{}] creating environment".format(FILE))
    gym.logger.set_level(gym.logger.ERROR)
    env = gym.make('CarRacing-v1').env
    state = env.reset()
    q, u, F = [], [], []
    done = False
    t = 0

    # load model
    model = load_model('models/dynamics_{}.h5'.format(args.model))

    # initialize controller
    controller = initialize_controller(args.control, model, env)

    # run simulation
    print("[{}] running simulation (t = {})".format(FILE, t))
    while not done:
        env.render()
        action, done = controller.step(state, t)
        q.append(np.array(state))
        u.append(np.array(action))
        state = env.step(action)
        F.append(np.array(state))
        t += 1
    print("[{}] stopping simulation (t = {})".format(FILE, t))

    # close environment
    print("[{}] closing environment".format(FILE))
    env.close()

    # plot trajectory
    if args.control == 'robot':
        plot_trajectory(env.track, F)
    else:
        plot_trajectory(env.track, F, model=model, data=[q, u], N=10)

    # save dataset
    if args.dataset is not None:
        save_dataset(args.dataset, np.array(q), np.array(u), np.array(F))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-d', '--dataset', metavar='dataset')
    parser.add_argument('-m', '--model', metavar='model', default='nonlinear')
    parser.add_argument('-c', '--control', metavar='control', default='robot')
    args = parser.parse_args()
    main(args)
