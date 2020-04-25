#!/usr/bin/env python

import os
import time
import argparse

import gym
import numpy as np
import matplotlib.pyplot as plt
from keras import models, layers

FILE = os.path.basename(__file__)
DIRECTORY = os.path.dirname(__file__)

LINE_WIDTH = 98

class Agent():
    def __init__(self, path, env, capacity=1000):
        print("[{}] initializing agent".format(FILE))

        if not os.path.exists(path):
            n = env.observation_space.shape[0] - 5
            m = env.action_space.shape[0]
            self.model = build_model(n, m)
        else:
            self.model = load_model(path)

        self.dt = env.dt
        self.path = path
        self.capacity = capacity

    def __del__(self):
        print("[{}] deinitializing agent".format(FILE))
        save_model(self.path, self.model)
        
    def train(self, samples, t):
        if int(t/self.dt) % (self.capacity//20) == 0:
            states, actions, observations = map(np.array, samples)

            # only use recent samples
            states = states[-self.capacity:]
            actions = actions[-self.capacity:]
            observations = observations[-self.capacity:]

            # format dataset
            dataset = format_dataset((states[::-1], actions[::-1], observations[::-1]), self.dt)

            # train model
            results = train_model(self.model, dataset, split=0.75)

    def dynamics(self, z, u):
        f = self.model.predict([z.reshape(1, -1), u.reshape(1, -1)])[0]
        F = np.zeros((6,))

        F[:3] = z[:3] + z[3:]*self.dt
        F[2] = F[2] % (2*np.pi)
        F[3:] = z[3:] + f*self.dt

        return F

    def horizon(self, z, u, H):
        z = np.vstack((z, np.zeros((H, z.shape[0]))))

        for i in range(H):
            z[i+1] = self.dynamics(z[i], u[i])
        
        return z[1:]

def save_dataset(path, dataset):
    print("[{}] saving dataset ({})".format(FILE, path))
    states, actions, observations = map(np.array, dataset)

    if os.path.exists(path):
        contents = np.load(path)
        states = np.vstack((contents['states'], states))
        actions = np.vstack((contents['actions'], actions))
        observations = np.vstack((contents['observations'], observations))

    np.savez_compressed(path, states=states, actions=actions, observations=observations)
    print("[{}] dataset saved ({} samples)".format(FILE, states.shape[0]))

def load_dataset(path, shuffle=False):
    print("[{}] loading dataset ({})".format(FILE, path))
    contents = np.load(path)
    states, actions, observations = contents['states'], contents['actions'], contents['observations']

    if shuffle is True:
        p = np.random.permutation(states.shape[0])
        states, actions, observations = states[p], actions[p], observations[p]

    print("[{}] dataset loaded ({} samples)".format(FILE, states.shape[0]))
    return states, actions, observations

def format_dataset(dataset, dt):
    states, actions, observations = map(np.array, dataset)
    z = states[:, :6]
    u = actions
    f = (observations[:, 3:6] - states[:, 3:6]) / dt
    return z, u, f

def save_model(path, model):
    print("[{}] saving model ({})".format(FILE, path))
    model.save(path)

def load_model(path):
    print("[{}] loading model ({})".format(FILE, path))
    model = models.load_model(path)
    
    print("{}".format('_' * LINE_WIDTH))
    model.summary()
    print("")

    return model

def build_model(n, m):
    print("[{}] building model".format(FILE))

    z = layers.Input(shape=(n,), name='z')
    u = layers.Input(shape=(m,), name='u')
    
    input_layer = layers.Concatenate(name ='input_layer')([z, u])
    hidden_layer_1 = layers.Dense(32, activation='tanh', name='hidden_layer_1')(input_layer)
    hidden_layer_2 = layers.Dense(32, activation='tanh', name='hidden_layer_2')(hidden_layer_1)
    output_layer = layers.Dense(n//2, activation='linear', name='output_layer')(hidden_layer_2)

    f = layers.Reshape((n//2,), name='f')(output_layer)

    model = models.Model(inputs=[z, u], outputs=f, name='dynamics')
    model.compile(loss='mean_squared_error', optimizer='rmsprop')

    print("{}".format('_' * LINE_WIDTH))
    model.summary()
    print("")

    return model

def train_model(model, dataset, split=0.25, batch_size=32, epochs=10, verbose=True):
    z, u, f = map(np.array, dataset)

    if verbose:
        print("{}".format('_' * LINE_WIDTH))

    # record training progress
    start = time.time()
    history = model.fit([z, u], f, validation_split=split, batch_size=batch_size, epochs=epochs, verbose=verbose)
    end = time.time()

    # calculate training time
    minutes, seconds = divmod(end-start, 60)

    if verbose:
        print("{}\n".format('_' * LINE_WIDTH))
    
    print("[{}] model trained ({}m {}s)".format(FILE, int(minutes), int(seconds)))
    return history.history

def plot_training(results):
    print("[{}] plotting training results".format(FILE))
    plt.figure(num='training')
    epochs = range(len(results['loss']))

    # plot training loss
    plt.plot(epochs, results['loss'], '-', label='Training') 
    plt.plot(epochs, results['val_loss'], '--', label='Validation')
    plt.title('Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid()
    
    plt.show()

def main(args):
    assert os.path.exists(args.dataset) and os.path.splitext(args.dataset)[1] == '.npz'
    assert os.path.splitext(args.model)[1] == '.h5'

    # get discretization time
    gym.logger.set_level(gym.logger.ERROR)
    dt = gym.make('CarRacing-v1').env.dt

    # load dataset
    states, actions, observations = load_dataset(args.dataset, shuffle=True)

    # format dataset
    dataset = format_dataset((states, actions, observations), dt)

    # build model
    model = build_model(n=6, m=3)

    # train model
    results = train_model(model, dataset, epochs=args.epochs)

    # plot training results
    plot_training(results)

    # save model
    save_model(args.model, model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-d', '--dataset', metavar='dataset', default='datasets/dynamics.npz')
    parser.add_argument('-m', '--model', metavar='model', default='models/dynamics.h5')
    parser.add_argument('-e', '--epochs', metavar='epochs', default=100, type=int)
    args = parser.parse_args()
    main(args)
