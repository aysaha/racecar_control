#!/usr/bin/env python

import os
import argparse

import gym
import numpy as np
import matplotlib.pyplot as plt

import controllers
import learning
from utils import *

FILE = os.path.basename(__file__)
DIRECTORY = os.path.dirname(__file__)

GRAY = '#666666'
ORANGE = '#CC6600'
BLUE = '#0066CC'

def plot_trajectory(track, car, model=None, data=None, N=1):
    print("[{}] plotting trajectory".format(FILE))

    # create plot title
    title = 'State Trajectory'

    # plot track
    track = np.vstack((track, track[0]))
    plt.plot(track[:, 2], track[:, 3], '-', linewidth=16, color=GRAY, label="Track")

    # plot car
    car = np.array(car)
    plt.plot(car[:, 0], car[:, 1], '-', linewidth=2, color=ORANGE, label="Car")
    
    if model is not None and data is not None:
        title += ' (N = {})'.format(N)
        data = np.array(data)
        samples = data.shape[0]
        q, u = data[:, :6], data[:, 6:]
        init = []
        pred = []

        # run model over prediction horizon
        for i in range(samples // N):
            if N > 1:
                init.append(q[N*i])
                pred.append(q[N*i])
            
            pred.extend(list(horizon(q[N*i], u[N*i:N*(i+1)], model, N)))

        # plot initial values
        if N > 1:
            init = np.array(init)
            plt.plot(init[:, 0], init[:, 1], 'o', markersize=4, color=BLUE)
        
        # plot predicted values
        pred = np.array(pred)
        plt.plot(pred[:, 0], pred[:, 1], '--', linewidth=2, color=BLUE, label="Model")

    # format plot
    plt.grid(which='major', color='#CCCCCC')
    plt.grid(which='minor', color='#DDDDDD')
    plt.minorticks_on()
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(borderpad=1.2, handletextpad=1.2)
    plt.show()

def main(args):
    assert args.dataset is None or os.path.splitext(args.dataset)[1] == '.npz'
    assert os.path.exists(args.model) and os.path.splitext(args.model)[1] == '.h5'
    assert args.control in ['robot', 'keyboard', 'xbox']
    
    # set logging level
    gym.logger.set_level(gym.logger.ERROR)

    # create environment
    print("[{}] creating environment".format(FILE))
    env = gym.make('CarRacing-v1').env
    state = env.reset()
    data, labels= [], []
    done = False
    t = 0

    # load dataset
    if args.dataset is not None and os.path.exists(args.dataset):
        saved_data, saved_labels = learning.load_dataset(args.dataset)
    else:
        saved_data, saved_labels = [], []

    # load model
    model = learning.load_model(args.model)

    # initialize controller
    controller = controllers.initialize_controller(args.control, model, env)

    # run simulation
    print("[{}] running simulation (t = {})".format(FILE, t))
    while not done:
        env.render()
        action, done = controller.step(state)
        data.append(np.concatenate((state, action)))
        state = env.step(action)
        labels.append(np.array(state))
        t += 1
    print("[{}] stopping simulation (t = {})".format(FILE, t))

    # close environment
    print("[{}] closing environment".format(FILE))
    env.close()

    # save dataset
    if args.dataset is not None:
        learning.save_dataset(args.dataset, saved_data + data, saved_labels + labels)

    # plot trajectory
    if args.control == 'robot':
        plot_trajectory(env.track, labels)
    else:
        plot_trajectory(env.track, labels, model=model, data=data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-d', '--dataset', metavar='dataset')
    parser.add_argument('-m', '--model', metavar='model', default='models/dynamics.h5')
    parser.add_argument('-c', '--control', metavar='control', default='robot')
    args = parser.parse_args()
    main(args)
