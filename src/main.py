#!/usr/bin/env python3

import os
import argparse

import gym
import numpy as np
from keras import models

from controllers import RobotController, KeyboardController, XboxController
from learning import load_dataset, save_dataset

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

def create_trajectory(track):
    trajectory = []

    for i, tile in enumerate(track):
        alpha, beta, x, y = tile
        waypoint = np.array([x, y, alpha, beta])
        trajectory.append(waypoint)

    return trajectory

def get_controller(ctrl, env):
    if ctrl == 'robot':
        controller = RobotController(env.viewer.window)
    elif ctrl == 'keyboard':
        controller = KeyboardController(env.viewer.window)
    elif ctrl == 'xbox':
        controller = XboxController()

    return controller

def main(args):
    assert args.controller in ['robot', 'keyboard', 'xbox']
    assert args.file is None or os.path.splitext(args.file)[1] == '.npz'

    # set logging level
    gym.logger.set_level(gym.logger.ERROR)

    # create environment
    print("[{}] creating environment".format(FILE))
    env = gym.make('CarRacing-v1').env
    state, track = env.reset()

    # create trajectory
    trajectory = create_trajectory(track)

    # initialize controller
    controller = get_controller(args.controller, env)

    # initialize dataset
    if args.file is not None and os.path.exists(args.file):
        data, labels = load_dataset(args.file)
    else:
        data, labels = [], []

    #model = models.load_model('models/dynamics.h5')

    print("[{}] running environment".format(FILE))
    done = False
    t = 0

    while not done:
        # display environment
        env.render()

        # control logic
        action, done = controller.step(state)

        #pred = model.predict(np.concatenate((state, action)).reshape((1,-1)))[0]

        # perform action and record dynamics
        data.append(np.concatenate((state, action)))
        state = env.step(action)
        labels.append(state)

        #if t % 10 == 0:
        #    print('--------------------')
        #    print_state(state)
        #    print_state(pred)

        # increase time
        t += 1

    # close environment
    print("[{}] closing environment".format(FILE))
    env.close()

    # save dataset
    if args.file is not None:
        save_dataset(args.file, data, labels)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('controller', help='robot, keyboard, or xbox')
    parser.add_argument('-f', '--file', metavar='path', help='file to save recorded data')
    args = parser.parse_args()
    main(args)
