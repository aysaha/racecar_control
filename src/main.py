#!/usr/bin/env python

import os
import argparse

import gym
import numpy as np
import matplotlib.pyplot as plt

from learning import horizon, load_model, save_dataset
from planning import plan_trajectory
from controllers import initialize_controller

FILE = os.path.basename(__file__)
DIRECTORY = os.path.dirname(__file__)

TRACK_COLOR = '#666666'
CAR_COLOR = '#CC6600'
MODEL_COLOR = '#0066CC'
MAJOR_GRID_COLOR = '#CCCCCC'
MINOR_GRID_COLOR = '#DDDDDD'

def plot_simulation(env, car, model=None, data=None, H=1):
    print("[{}] plotting trajectory".format(FILE))
    plt.figure(num='simulation')

    # create plot title
    title = 'Simulation'

    # plot track
    track = np.vstack((env.track, env.track[0]))
    plt.plot(track[:, 2], track[:, 3], '-', linewidth=16, color=TRACK_COLOR, label="Track")

    # plot car
    car = np.array(car)
    plt.plot(car[:, 0], car[:, 1], '-', linewidth=2, color=CAR_COLOR, label="Car")

    # plot model
    if model is not None and data:
        title += ' (H = {})'.format(H)
        z, u = np.array(data[0]), np.array(data[1])
        init, pred = [], []

        # run model over prediction horizon
        for i in range(z.shape[0]//H):
            if H > 1:
                init.append(z[H*i])
                pred.append(z[H*i])
            
            pred.extend(list(horizon(z[H*i], u[H*i:H*(i+1)], env.dt, model, H)))

        # plot initial values
        if H > 1:
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
    assert os.path.splitext(args.dataset)[1] == '.npz'
    assert os.path.exists(args.model) and os.path.splitext(args.model)[1] == '.h5'
    assert args.control in ['robot', 'keyboard', 'xbox']

    # create environment
    print("[{}] creating environment".format(FILE))
    gym.logger.set_level(gym.logger.ERROR)
    env = gym.make('CarRacing-v1').env
    state = env.reset()
    z, u, F = [], [], []
    done = False

    # load model
    model = load_model(args.model)

    # plan trajectory
    trajectory = plan_trajectory(env)

    # initialize controller
    controller = initialize_controller(args.control, trajectory, model, env)

    # run simulation
    print("[{}] running simulation (t = {}s)".format(FILE, int(env.t)))
    while not done:
        try:
            # render environment
            env.render()

            # perform a simulation step and save dynamics
            action, done = controller.step(state, env.t)
            z.append(np.array(state))
            u.append(np.array(action))
            state = env.step(action)
            F.append(np.array(state))
        except KeyboardInterrupt:
            break
    print("[{}] stopping simulation (t = {}s)".format(FILE, int(env.t)))

    # close environment
    print("[{}] closing environment".format(FILE))
    env.close()

    # plot simulation results
    if args.control == 'robot':
        plot_simulation(env, F)
    else:
        plot_simulation(env, F, model=model, data=[z, u], H=50)

    # save dataset
    save_dataset(args.dataset, z, u, F)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-d', '--dataset', metavar='dataset', default='datasets/dynamics.npz')
    parser.add_argument('-m', '--model', metavar='model', default='models/dynamics.h5')
    parser.add_argument('-c', '--control', metavar='control', default='robot')
    args = parser.parse_args()
    main(args)
