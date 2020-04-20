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

def dynamics(z, u, dt, model):
    assert z.shape == (6,) or z.shape == (6, 1)
    assert u.shape == (3,) or u.shape == (3, 1)
    
    tensor = np.hstack((z, u))
    layers = model.get_weights()
    L = len(layers)
    F = np.zeros((6,))

    F[:3] = z[:3] + z[3:]*dt
    F[2] = F[2] % (2*np.pi)

    for i in range(0, L, 2):
        if i < L-2:
            tensor = np.tanh(layers[i].T @ tensor  + layers[i+1])
        else:
            tensor = layers[i].T @ tensor  + layers[i+1]

    f = tensor
    F[3:] = z[3:] + f*dt

    return F

def horizon(z, u, dt, model, H):
    assert z.shape == (6,) or z.shape == (6, 1)
    assert u.shape == (H, 3)

    z = np.vstack((z, np.zeros((H, z.shape[0]))))

    for i in range(H):
        z[i+1] = dynamics(z[i], u[i], dt, model)
    
    return z[1:]

class Agent():
    def __init__(self, path, env, reset=False, capacity=10000):
        print("[{}] initializing agent".format(FILE))

        if reset or not os.path.exists(path):
            n = env.observation_space.shape[0] - 5
            m = env.action_space.shape[0]
            self.model = build_model(n, m)
        else:
            self.model = load_model(path)

        self.dt = env.dt
        self.path = path
        self.capacity = capacity
        self.results = None

    def __del__(self):
        print("[{}] deinitializing agent".format(FILE))
        save_model(self.path, self.model)
        
    def train(self, samples):
        states, actions, observations = map(np.array, samples)

        # format dataset
        states = states[-self.capacity:]
        actions = actions[-self.capacity:]
        observations = observations[-self.capacity:]
        dataset = (states[::-1], actions[::-1], observations[::-1])

        # train model
        self.results = train_model(self.model, dataset, self.dt)

    def predict(self):
        pass

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

def save_model(path, model):
    print("[{}] saving model ({})".format(FILE, path))
    model.save(path)

def load_model(path):
    print("[{}] loading model ({})".format(FILE, path))
    model = models.load_model(path)
    return model

def build_model(n, m):
    print("[{}] building model".format(FILE))

    z = layers.Input(shape=(n,), name='z')
    u = layers.Input(shape=(m,), name='u')
    
    input_layer = layers.Concatenate(name ='input_layer')([z, u])
    hidden_layer_1 = layers.Dense(16, activation='tanh', name='hidden_layer_1')(input_layer)
    hidden_layer_2 = layers.Dense(16, activation='tanh', name='hidden_layer_2')(hidden_layer_1)
    output_layer = layers.Dense(n//2, activation='linear', name='output_layer')(hidden_layer_2)

    f = layers.Reshape((n//2,), name='f')(output_layer)

    model = models.Model(inputs=[z, u], outputs=f, name='dynamics')
    model.compile(loss='mse', optimizer='rmsprop')

    return model

def train_model(model, dataset, dt, split=0.75, batch_size=32, epochs=10, verbose=False):
    states, actions, observations = map(np.array, dataset)

    if verbose:
        print("{}".format('_' * 98))
        model.summary()

    # format data
    z = states[:, :6]
    u = actions
    f = (observations[:, 3:6] - states[:, 3:6]) / dt

    # record training progress
    start = time.time()
    history = model.fit([z, u], f, validation_split=split, batch_size=batch_size, epochs=epochs, verbose=verbose)
    end = time.time()

    # calculate training time
    minutes, seconds = divmod(end-start, 60)

    if verbose:
        print("{}\n".format('_' * 98))
    
    print("[{}] trained model ({}m {}s)".format(FILE, int(minutes), int(seconds)))
    return history.history

def plot_training(results):
    print("[{}] plotting training results".format(FILE))
    plt.figure(num='training')
    epochs = np.arange(len(results['loss']))

    # plot training loss
    plt.plot(epochs, results['loss'], '-', label='Training') 
    plt.plot(epochs, results['val_loss'], '--', label='Validation')
    plt.title('Training Results')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    
    plt.show()

def main(args):
    assert os.path.exists(args.dataset) and os.path.splitext(args.dataset)[1] == '.npz'
    assert os.path.splitext(args.model)[1] == '.h5'

    # create environment
    gym.logger.set_level(gym.logger.ERROR)
    env = gym.make('CarRacing-v1').env

    # load dataset
    dataset = load_dataset(args.dataset, shuffle=True)

    # create agent
    agent = Agent(args.model, env, reset=True)

    # train model
    results = train_model(agent.model, dataset, agent.dt, split=0.25, epochs=args.epochs, verbose=True)

    # plot training results
    plot_training(results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-d', '--dataset', metavar='dataset', default='datasets/dynamics.npz')
    parser.add_argument('-m', '--model', metavar='model', default='models/dynamics.h5')
    parser.add_argument('-e', '--epochs', metavar='epochs', default=100, type=int)
    args = parser.parse_args()
    main(args)
