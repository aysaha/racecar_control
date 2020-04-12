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

def print_state(state):
    string = ""
    string += "x = {:7.2f} m | ".format(state[0])
    string += "y = {:7.2f} m | ".format(state[1])
    string += "theta = {:4.2f} rad | ".format(state[2])
    string += "v_x = {:7.2f} m/s | ".format(state[3])
    string += "v_y = {:7.2f} m/s | ".format(state[4])
    string += "omega = {:6.2f} rad".format(state[5]) 
    print(string)

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
    assert args.control in ['robot', 'keyboard', 'xbox']
    assert args.dataset is None or os.path.splitext(args.dataset)[1] == '.npz'
    assert args.model is None or os.path.splitext(args.model)[1] == '.h5'

    # set logging level
    gym.logger.set_level(gym.logger.ERROR)

    # create environment
    print("[{}] creating environment".format(FILE))
    env = gym.make('CarRacing-v1').env
    state, track = env.reset()

    # initialize controller
    controller = controllers.get_controller(args.control, env)

    # initialize dataset
    if args.dataset is not None and os.path.exists(args.dataset):
        saved_data, saved_labels = learning.load_dataset(args.dataset)
    else:
        saved_data, saved_labels = [], []

    # initialize model
    if args.model is not None:
        model = learning.load_model(args.model)
    else:
        model = None

    # start environment
    print("[{}] running environment".format(FILE))
    data, labels= [], []
    done = False
    t = 0

    while not done:
        # display environment
        env.render()

        # control logic
        action, done = controller.step(state)

        # perform action and record dynamics
        data.append(np.concatenate((state, action)))
        state = env.step(action)
        labels.append(np.array(state))

        # increment time
        t += 1

    # close environment
    print("[{}] closing environment".format(FILE))
    env.close()

    # save dataset
    if args.dataset is not None:
        learning.save_dataset(args.dataset, saved_data + data, saved_labels + labels)

    # visualize trajectory
    plot_trajectory(track, labels, model=model, data=data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-c', '--control', metavar='control', default='keyboard')
    parser.add_argument('-d', '--dataset', metavar='dataset', default='data/dynamics.npz')
    parser.add_argument('-m', '--model', metavar='model', default='models/dynamics.h5')
    args = parser.parse_args()
    main(args)
