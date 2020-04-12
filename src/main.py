#!/usr/bin/env python3

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
    assert args.controller in ['robot', 'keyboard', 'xbox']
    assert args.input is None or os.path.splitext(args.input)[1] == '.h5'
    assert args.output is None or os.path.splitext(args.output)[1] == '.npz'

    # set logging level
    gym.logger.set_level(gym.logger.ERROR)

    # create environment
    print("[{}] creating environment".format(FILE))
    env = gym.make('CarRacing-v1').env
    state, track = env.reset()

    # initialize controller
    controller = controllers.get_controller(args.controller, env)

    # initialize model
    if args.input is not None:
        model = learning.load_model(args.input)
    else:
        model = None

    # initialize dataset
    if args.output is not None and os.path.exists(args.output):
        saved_data, saved_labels = learning.load_dataset(args.output)
    else:
        saved_data, saved_labels = [], []

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
    if args.output is not None:
        learning.save_dataset(args.output, saved_data + data, saved_labels + labels)

    # visualize trajectory
    plot_trajectory(track, labels, model=model, data=data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('controller')
    parser.add_argument('-i', '--input', metavar='model')
    parser.add_argument('-o', '--output', metavar='dataset')
    args = parser.parse_args()
    main(args)
