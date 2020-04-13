#!/usr/bin/env python

import os
import argparse

import gym
import numpy as np
import matplotlib.pyplot as plt

import controllers
import learning

FILE = os.path.basename(__file__)
DIRECTORY = os.path.dirname(__file__)

def plot_trajectory(track, car, model=None, data=None):
    print("[{}] plotting trajectory".format(FILE))

    # plot track
    track = np.vstack((track, track[0]))
    plt.plot(track[:, 2], track[:, 3], '-', linewidth=16, color='#666666', label="Track")

    # plot car
    car = np.array(car)
    plt.plot(car[:, 0], car[:, 1], '-', linewidth=2, color='#CC6600', label="Car")
    
    # plot model
    if model is not None and data is not None:
        pred = model.predict(np.array(data))
        plt.plot(pred[:, 0], pred[:, 1], '--',linewidth=2, color='#0066CC', label="Model")
    
    # format plot
    plt.grid(which='major', color='#CCCCCC')
    plt.grid(which='minor', color='#DDDDDD')
    plt.minorticks_on()
    plt.title('State Trajectory')
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
    controller = controllers.get_controller(args.control, model, env)

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
    plot_trajectory(env.track, labels, model=model, data=data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-d', '--dataset', metavar='dataset')
    parser.add_argument('-m', '--model', metavar='model', default='models/dynamics.h5')
    parser.add_argument('-c', '--control', metavar='control', default='keyboard')
    args = parser.parse_args()
    main(args)
