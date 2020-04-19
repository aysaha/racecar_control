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

class Agent():
    def __init__(self, path, reset=False, capacity=10000):
        print("[{}] initializing agent".format(FILE))

        if reset or not os.path.exists(path):
            self.model = build_model(3, 3)
        else:
            self.model = load_model(path)

        self.buffer = {'z': [], 'u': [], 'F': []}
        self.path = path
        self.capacity = capacity

    def __del__(self):
        save_model(self.path, self.model)

    def save(self, sample):
        z, u, F = sample

        self.buffer['z'].append(np.array(z))
        self.buffer['u'].append(np.array(u))
        self.buffer['F'].append(np.array(F))

        while len(self.buffer['z']) > self.capacity: self.buffer['z'].pop(0)
        while len(self.buffer['u']) > self.capacity: self.buffer['u'].pop(0)
        while len(self.buffer['F']) > self.capacity: self.buffer['F'].pop(0)
        
    def train(self, dt, dataset=None, split=0.75, batch_size=32, epochs=10, verbose=False):
        print("[{}] training model ({} samples)".format(FILE, len(self.buffer['z'])))

        if dataset:
            z, u, F = dataset
        else:
            z, u, F = np.array(self.buffer['z']), np.array(self.buffer['u']), np.array(self.buffer['F'])
        
        # shuffle data
        p = np.random.permutation(z.shape[0])
        z, u, F = z[p], u[p], F[p]

        # format data
        z = z[:, :6]
        f = (F[:, 3:6] - z[:, 3:6]) / dt

        if verbose:
            print("{}".format('_' * 98))

        history = self.model.fit([z, u], f,
                                 validation_split=split,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 verbose=verbose)
        
        if verbose:
            print("{}\n".format('_' * 98))

        return history.history

    def predict(self):
        pass

def dynamics(z, u, dt, model):
    assert z.shape == (11,) or z.shape == (11, 1)
    assert u.shape == (3,) or u.shape == (3, 1)
    
    tensor = np.hstack((z, u))
    layers = model.get_weights()
    L = len(layers)
    F = np.zeros((11,))

    F[:3] = z[:3] + z[3:6]*dt
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
    assert z.shape == (11,) or z.shape == (11, 1)
    assert u.shape == (H, 3)

    z = np.vstack((z, np.zeros((H, z.shape[0]))))

    for i in range(H):
        z[i+1] = dynamics(z[i], u[i], dt, model)
    
    return z[1:]

def save_dataset(path, z, u, F, limit=100000):
    print("[{}] saving dataset ({})".format(FILE, path))
    z, u, F = np.array(z), np.array(u), np.array(F)

    if os.path.exists(path):
        contents = np.load(path)
        z = np.vstack((contents['z'], z))
        u = np.vstack((contents['u'], u))
        F = np.vstack((contents['F'], F))

    if z.shape[0] > limit:
        p = np.random.permutation(limit)
        z, u, F = z[p], u[p], F[p]

    np.savez_compressed(path, z=z, u=u, F=F)
    print("[{}] dataset saved ({} samples)".format(FILE, z.shape[0]))

def load_dataset(path, shuffle=False):
    print("[{}] loading dataset ({})".format(FILE, path))
    contents = np.load(path)
    z, u, F = contents['z'], contents['u'], contents['F']

    if shuffle is True:
        p = np.random.permutation(z.shape[0])
        z, u, F = z[p], u[p], F[p]

    print("[{}] dataset loaded ({} samples)".format(FILE, z.shape[0]))
    return z, u, F

def save_model(path, model):
    print("[{}] saving model ({})".format(FILE, path))
    model.save(path)

def load_model(path):
    print("[{}] loading model ({})".format(FILE, path))
    model = models.load_model(path)
    return model

def build_model(n, m):
    print("[{}] building model".format(FILE))

    z = layers.Input(shape=(2*n,), name='z')
    u = layers.Input(shape=(m,), name='u')
    
    input_layer = layers.Concatenate(name ='input_layer')([z, u])
    hidden_layer_1 = layers.Dense(16, activation='tanh', name='hidden_layer_1')(input_layer)
    hidden_layer_2 = layers.Dense(16, activation='tanh', name='hidden_layer_2')(hidden_layer_1)
    output_layer = layers.Dense(n, activation='linear', name='output_layer')(hidden_layer_2)

    f = layers.Reshape((n,), name='f')(output_layer)

    model = models.Model(inputs=[z, u], outputs=f, name='dynamics')
    model.compile(loss='mse', optimizer='rmsprop', metrics=['acc'])

    return model

def train_model(model, z, u, F, dt, batch_size=32, epochs=10, verbose=False):
    print("[{}] training model ({} samples)".format(FILE, z.shape[0]))

    if verbose:
        print("{}".format('_' * 98))
        model.summary()

    # format training data
    # z = [q_n q_dot_n q_bar_n]
    # F = [q_n+1 q_dot_n+1 q_bar_n+1]
    z = z[:, :6]
    f = (F[:, 3:6] - z[:, 3:6]) / dt

    # record training progress
    start = time.time()
    history = model.fit([z, u], f, batch_size=batch_size, epochs=epochs, verbose=verbose)
    end = time.time()

    if verbose:
        minutes, seconds = divmod(end-start, 60)
        print("{}\n".format('_' * 98))
        print("[{}] training completed ({}m {}s)".format(FILE, int(minutes), int(seconds)))
    
    return history.history

def plot_training(results):
    print("[{}] plotting training results".format(FILE))
    plt.figure(num='training')
    epochs = np.arange(len(results['loss']))

    # plot training loss
    plt.subplot(2, 1, 1)  
    plt.plot(epochs, results['loss'])
    plt.title('Training')
    plt.ylabel('Loss')
    plt.grid()

    # plot training accuracy
    plt.subplot(2, 1, 2)  
    plt.plot(epochs, results['acc'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid()

    plt.show()

def main(args):
    assert os.path.exists(args.dataset) and os.path.splitext(args.dataset)[1] == '.npz'
    assert os.path.splitext(args.model)[1] == '.h5'

    # get discretization time
    gym.logger.set_level(gym.logger.ERROR)
    dt = gym.make('CarRacing-v1').env.dt

    # load dataset
    dataset = load_dataset(args.dataset)

    # create agent
    agent = Agent(args.model, reset=True)

    # train model
    results = agent.train(dt, dataset=dataset, split=0, epochs=args.epochs, verbose=True)

    # plot training results
    plot_training(results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-d', '--dataset', metavar='dataset', default='datasets/dynamics.npz')
    parser.add_argument('-m', '--model', metavar='model', default='models/dynamics.h5')
    parser.add_argument('-e', '--epochs', metavar='epochs', default=100, type=int)
    args = parser.parse_args()
    main(args)
