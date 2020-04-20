#!/usr/bin/env python

import os
import argparse

import gym
import numpy as np
import matplotlib.pyplot as plt

from learning import Agent, save_dataset
from planning import plan_trajectory
from controllers import initialize_controller

FILE = os.path.basename(__file__)
DIRECTORY = os.path.dirname(__file__)

TRACK_COLOR = '#666666'
CAR_COLOR = '#CC6600'
MODEL_COLOR = '#0066CC'
MAJOR_GRID_COLOR = '#CCCCCC'
MINOR_GRID_COLOR = '#DDDDDD'

def plot_simulation(car,  env, model=None, data=None, H=1):
    print("[{}] plotting simulation".format(FILE))
    plt.figure(num='simulation')

    # create plot title
    title = 'Simulation Results'

    # plot track
    track = np.vstack((env.track, env.track[0]))
    plt.plot(track[:, 2], track[:, 3], '-', linewidth=16, color=TRACK_COLOR, label="Track")

    # plot car
    car = np.array(car)
    plt.plot(car[:, 0], car[:, 1], '-', linewidth=2, color=CAR_COLOR, label="Car")

    # plot model
    if model is not None and data:
        title += ' (H = {})'.format(H)
        states, actions = map(np.array, data)
        init, pred = [], []

        '''
        # run model over prediction horizon
        for i in range(states.shape[0]//H):
            if H > 1:
                init.append(z[H*i])
                pred.append(z[H*i])
            
            pred.extend(list(horizon(z[H*i], u[H*i:H*(i+1)], env.dt, model, H)))

        # plot initial values
        if H > 1:
            init = np.array(init)
            plt.plot(init[:, 0], init[:, 1], 'o', markersize=4, color=MODEL_COLOR)
        '''

        z = states[:, :6]
        u = actions
        f = model.predict([z, u])
        q_dot = states[:, 3:6] + f*env.dt
        q = states[:, :3] + q_dot*env.dt
        pred = q

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
    T = 150

    # plan trajectory
    trajectory = plan_trajectory(env)

    # initialize controller
    controller = initialize_controller(args.control, trajectory, env)

    # initialize agent
    agent = Agent(args.model, env)

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
            agent.save((state, action, observation))

            if args.control == 'robot' and int(env.t/env.dt) % int(T/env.dt) == 0:
                agent.train(env.t)

            # update current state
            state = observation
        except KeyboardInterrupt:
            break
    print("[{}] stopping simulation (t = {}s)".format(FILE, int(env.t)))

    # close environment
    print("[{}] closing environment".format(FILE))
    env.close()

    # plot simulation results
    if args.control == 'robot':
        plot_simulation(observations, env)
    else:
        plot_simulation(observations, env, model=agent.model, data=(states, actions), H=1)

    # save dataset
    if args.control == 'xbox':
        save_dataset(args.dataset, (states, actions, observations))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-d', '--dataset', metavar='dataset', default='datasets/bootstrap.npz')
    parser.add_argument('-m', '--model', metavar='model', default='models/dynamics.h5')
    parser.add_argument('-c', '--control', metavar='control', default='robot')
    args = parser.parse_args()
    main(args)
