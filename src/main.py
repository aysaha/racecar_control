#!/usr/bin/env python

import os
import argparse

import gym
import numpy as np
import matplotlib.pyplot as plt

from learning import Agent, save_dataset
from controllers import initialize_controller

FILE = os.path.basename(__file__)
DIRECTORY = os.path.dirname(__file__)

TRACK_COLOR = '#666666'
CAR_COLOR = '#CC6600'
MODEL_COLOR = '#0066CC'
MAJOR_GRID_COLOR = '#CCCCCC'
MINOR_GRID_COLOR = '#DDDDDD'

def plot_simulation(car,  env, agent=None, data=None, H=1):
    print("[{}] plotting simulation".format(FILE))
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
    if agent is not None and data:
        title += ' (H = {})'.format(H)
        states, actions = map(np.array, data)
        z, u = states[:, :6], actions
        init, pred = [], []

        # run model over prediction horizon
        for i in range(states.shape[0]//H):
            if H > 1:
                init.append(z[H*i])
                pred.append(z[H*i])
            
            pred.extend(list(agent.horizon(z[H*i], u[H*i:H*(i+1)], H)))

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
    assert os.path.splitext(args.model)[1] == '.h5'
    assert args.control in ['robot', 'keyboard', 'xbox']

    # create environment
    print("[{}] creating environment".format(FILE))
    gym.logger.set_level(gym.logger.ERROR)
    env = gym.make('CarRacing-v1').env
    state = env.reset()
    states, actions, observations = [], [], []
    done = False

    # initialize agent
    agent = Agent(args.model, env)

    # initialize controller
    controller = initialize_controller(args.control, agent.model, env)

    # run simulation
    print("[{}] running simulation (t = {}s)".format(FILE, int(env.t)))
    while not done:
        try:
            # render environment
            env.render()

            # run control policy
            action, done = controller.step(state, env.t)

            # peform a simulation step
            observation = env.step(action)

            # save interaction
            states.append(np.array(state))
            actions.append(np.array(action))
            observations.append(np.array(observation))

            #if args.control == 'robot':
            #    agent.train((states, actions, observations), env.t)

            # update current state
            state = np.array(observation)
        except KeyboardInterrupt:
            done = True
    print("[{}] stopping simulation (t = {}s)".format(FILE, int(env.t)))

    # close environment
    print("[{}] closing environment".format(FILE))
    env.close()
    
    # plot simulation results
    if args.control == 'robot':
        plot_simulation(observations, env)
    else:
        plot_simulation(observations, env, agent=agent, data=(states, actions), H=16)

    # save dataset
    if args.control == 'xbox':
        save_dataset(args.dataset, (states, actions, observations))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-d', '--dataset', metavar='dataset', default='datasets/dynamics.npz')
    parser.add_argument('-m', '--model', metavar='model', default='models/dynamics.h5')
    parser.add_argument('-c', '--control', metavar='control', default='robot')
    args = parser.parse_args()
    main(args)
