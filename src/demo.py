#!/usr/bin/env python3

"""
Description: Demo program for CarRacing-v1 environment
Author: Ayusman Saha
"""
import argparse
import gym
import numpy as np
from pyglet.window import key

ACTION = np.array([0.0, 0.0, 0.0])
DONE = False    

def key_press(k, mod):
    global ACTION, DONE
    if k == key.Q:        DONE = True
    if k == key.LEFT:     ACTION[0] = -1.0
    if k == key.RIGHT:    ACTION[0] = +1.0
    if k == key.UP:       ACTION[1] = +1.0
    if k == key.DOWN:     ACTION[2] = +0.8

def key_release(k, mod):
    global ACTION
    if k == key.LEFT  and ACTION[0] == -1.0:    ACTION[0] = 0.0
    if k == key.RIGHT and ACTION[0] == +1.0:    ACTION[0] = 0.0
    if k == key.UP:                             ACTION[1] = 0.0
    if k == key.DOWN:                           ACTION[2] = 0.0

def print_state(state, raw=False):
    string = ""

    if raw is True:
        string += "{:f},".format(state[0])
        string += "{:f},".format(state[1])
        string += "{:f},".format(state[2])
        string += "{:f},".format(state[3])
        string += "{:f},".format(state[4])
        string += "{:f};".format(state[5]) 
    else:
        string += "x = {:7.2f} m | ".format(state[0])
        string += "y = {:7.2f} m | ".format(state[1])
        string += "theta = {:4.2f} rad | ".format(state[2])
        string += "v_x = {:7.2f} m/s | ".format(state[3])
        string += "v_y = {:7.2f} m/s | ".format(state[4])
        string += "omega = {:6.2f} rad".format(state[5]) 

    print(string)

def create_path(track):
    path = []

    # TODO - find a good representation for the path
    for i, tile in enumerate(track):
        alpha, beta, x, y = tile
        waypoint = np.array([x, y, alpha, beta])
        path.append(waypoint)

    return path

def main(args):
    global ACTION, DONE

    # set logging level
    gym.logger.set_level(gym.logger.ERROR)

    # create environment
    env = gym.make('CarRacing-v1', verbose=0).env

    # initialize environment
    state, track = env.reset()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    t = 0

    # create path
    path = create_path(track)

    while not DONE:
        # display environment
        env.render()
        
        # decode state
        x, y, theta, v_x, v_y, omega = state

        # control logic
        # TODO - implement controller
        action = ACTION

        # perform action
        state, reward, done, info = env.step(action)

        # display state
        if t % 100 == 0:
            print_state(state)
        
        # increase time
        t += 1

    # close environment
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
