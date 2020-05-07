#!/usr/bin/env python

import os
import time
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

class LapCounter():
    def __init__(self, env):
        self.start = env.track[1]
        self.delta = env.track_width
        self.t_sim = env.t
        self.t_wall = time.time()
        self.flag = False
        self.laps = []

    def update(self, state, t):
        # determine distance to starting point
        distance = np.linalg.norm(self.start[2:] - state[:2])

        if distance <= self.delta and self.flag is True:
            # calculate lap time
            t_lap = (t - self.t_sim, time.time() - self.t_wall)
            self.laps.append(t_lap)
            print("[{}] lap #{} completed in {:.3f}s ({:.3f}s)".format(FILE, len(self.laps), t_lap[0], t_lap[1]))

            # reset timers
            self.t_sim = t
            self.t_wall = time.time()
            self.flag = False
        elif distance > self.delta:
            self.flag = True

def plot_simulation(car,  env, agent=None, data=None, H=1):
    print("[{}] plotting simulation".format(FILE))
    plt.figure(num='simulation')

    # create plot title
    title = 'Simulation'

    # plot track
    track = np.vstack((env.track, env.track[0]))
    plt.plot(track[:, 2], track[:, 3], '-', linewidth=24, color=TRACK_COLOR, label="Track")

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
    plt.legend(loc='upper right', borderpad=1.5, labelspacing=1.5, handletextpad=2.5)

    # format grid
    plt.grid(which='major', color=MAJOR_GRID_COLOR)
    plt.grid(which='minor', color=MINOR_GRID_COLOR)
    plt.minorticks_on()

    plt.show()

def main(args):
    assert os.path.exists(args.model) and os.path.splitext(args.model)[1] == '.h5'
    assert os.path.splitext(args.dataset)[1] == '.npz'
    assert args.control in ['robot', 'keyboard', 'xbox']

    # create environment
    print("[{}] creating environment".format(FILE))
    gym.logger.set_level(gym.logger.ERROR)
    env = gym.make('CarRacing-v1', seed=args.seed).env

    # initialize environment
    print("[{}] initializing environment ({})".format(FILE, env.id))
    state = env.reset()
    states = []
    actions = []
    observations = []
    done = False

    # initialize agent
    agent = Agent(args.model, env)

    # initialize controller
    controller = initialize_controller(args.control, agent.model, env)

    # start lap counter
    counter = LapCounter(env)

    # run simulation
    print("[{}] running simulation (t = {}s)".format(FILE, int(env.t)))
    while not done:
        try:
            # render environment
            env.render()

            # update lap counter
            counter.update(state, env.t)

            # run control policy
            action, done = controller.step(state, env.t)

            # peform a simulation step
            observation = env.step(action)

            # save interaction
            states.append(np.array(state))
            actions.append(np.array(action))
            observations.append(np.array(observation))

            if args.control == 'robot':
                agent.train((states, actions, observations), env.t)

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
        plot_simulation(observations, env, agent=agent, data=(states, actions), H=10)

    # save dataset
    if args.control == 'xbox':
        save_dataset(args.dataset, (states, actions, observations))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-m', '--model', metavar='model', default='models/dynamics.h5')
    parser.add_argument('-d', '--dataset', metavar='dataset', default='datasets/dynamics.npz')
    parser.add_argument('-c', '--control', metavar='control', default='robot')
    parser.add_argument('-s', '--seed', metavar='seed', default=None, type=lambda x: int(x, 0))
    args = parser.parse_args()
    main(args)
