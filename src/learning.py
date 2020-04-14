#!/usr/bin/env python

import os
import time
import argparse

import numpy as np
import matplotlib.pyplot as plt
from keras import models, layers

FILE = os.path.basename(__file__)
DIRECTORY = os.path.dirname(__file__)

EPOCHS = 100
BATCH_SIZE = 32

def dynamics(q, u, model):
    assert q.shape == (6,) or q.shape == (6, 1)
    assert u.shape == (3,) or q.shape == (3, 1)
    assert model is not None

    q = q.reshape(1, -1)
    u = u.reshape(1, -1)
    F = model.predict([q, u])[0]

    return F

def horizon(q, u, model, N):
    assert q.shape == (6,) or q.shape == (6, 1)
    assert u.shape == (N, 3)
    assert model is not None
    assert N > 0

    q = np.vstack((q, np.zeros((N, q.shape[0]))))

    for i in range(N):
        q[i+1] = dynamics(q[i], u[i], model)
    
    return q[1:]

def save_dataset(path, q, u, F):
    print("[{}] saving dataset ({})".format(FILE, path))

    if os.path.exists(path):
        contents = np.load(path)
        q, u, F = np.vstack((contents['q'], q)), np.vstack((contents['u'], u)), np.vstack((contents['F'], F))

    np.savez_compressed(path, q=q, u=u, F=F)

def load_dataset(path, shuffle=False):
    print("[{}] loading dataset ({})".format(FILE, path))
    contents = np.load(path)
    q, u, F = contents['q'], contents['u'], contents['F']

    if shuffle is True:
        p = np.random.permutation(q.shape[0])
        q, u, F = q[p], u[p], F[p]

    return q, u, F

def save_model(path, model):
    print("[{}] saving model ({})".format(FILE, path))
    model.save(path)

def load_model(path):
    print("[{}] loading model ({})".format(FILE, path))
    return models.load_model(path)

def build_model(N, M, config, summary=False):
    # define input layers
    q = layers.Input(shape=(N,), name='q')
    u = layers.Input(shape=(M,), name='u')
    q_u = layers.Concatenate(name ='q_u')([q, u])

    if config == 'nonlinear':
        # define hidden layers
        hidden_layer_1 = layers.Dense(16, activation='relu', name='hidden_layer_1')(q_u)
        hidden_layer_2 = layers.Dense(16, activation='relu', name='hidden_layer_2')(hidden_layer_1)
        hidden_layer_3 = layers.Dense(N, activation='linear', name='hidden_layer_3')(hidden_layer_2)
        
        # define output layer
        F = layers.Reshape((N,), name='F')(hidden_layer_3)
    elif config == 'affine':
        # define hidden layers for f
        hidden_layer_1f = layers.Dense(8, activation='relu', name='hidden_layer_1f')(q)
        hidden_layer_2f = layers.Dense(8, activation='relu', name='hidden_layer_2f')(hidden_layer_1f)
        hidden_layer_3f = layers.Dense(N, activation='linear', name='hidden_layer_3f')(hidden_layer_2f)
        f = layers.Reshape((N,), name='f')(hidden_layer_3f)

        # define hidden layers for g
        hidden_layer_1g = layers.Dense(8, activation='relu', name='hidden_layer_1g')(q)
        hidden_layer_2g = layers.Dense(8, activation='relu', name='hidden_layer_2g')(hidden_layer_1g)
        hidden_layer_3g = layers.Dense(N*M, activation='linear', name='hidden_layer_3g')(hidden_layer_2g)
        g = layers.Reshape((N, M), name='g')(hidden_layer_3g)
        gu = layers.Dot(-1, name='gu')([g, u])  

        # define output layer
        F = layers.Add(name='F')([f, gu])
    elif config == 'linear':
        # define hidden layers for A
        hidden_layer_1A = layers.Dense(8, activation='relu', name='hidden_layer_1A')(q_u)
        hidden_layer_2A = layers.Dense(8, activation='relu', name='hidden_layer_2A')(hidden_layer_1A)
        hidden_layer_3A = layers.Dense(N*N, activation='linear', name='hidden_layer_3A')(hidden_layer_2A)
        A = layers.Reshape((N, N), name='A')(hidden_layer_3A)
        Aq = layers.Dot(-1, name='Aq')([A, q])

        # define hidden layers for B
        hidden_layer_1B = layers.Dense(8, activation='relu', name='hidden_layer_1B')(q_u)
        hidden_layer_2B = layers.Dense(8, activation='relu', name='hidden_layer_2B')(hidden_layer_1B)
        hidden_layer_3B = layers.Dense(N*M, activation='linear', name='hidden_layer_3B')(hidden_layer_2B)
        B = layers.Reshape((N, M), name='B')(hidden_layer_3B)
        Bu = layers.Dot(-1, name='Bu')([B, u])

        # define output layer
        F = layers.Add(name='F')([Aq, Bu])
    
    # define model
    model = models.Model(inputs=[q, u], outputs=F, name=config)
    model.compile(loss='mse', optimizer='rmsprop', metrics=['acc'])

    # show summary
    if summary is True:
        model.summary()

    return model

def plot_training(epochs, results):
    print("[{}] plotting training results".format(FILE))
    plt.subplots(2, num='training_results')

    # plot training loss
    plt.subplot(2, 1, 1)  
    plt.plot(epochs, results['loss'])
    plt.title('Training Results')
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
    assert args.config in ['nonlinear', 'affine', 'linear']

    # load dataset
    q, u, F = load_dataset(args.dataset, shuffle=True)

    # build model
    print("{}".format('_' * 98))
    model = build_model(q.shape[1], u.shape[1], config=args.config, summary=True)

    # train model
    start = time.time()
    history = model.fit([q, u], F, epochs=EPOCHS, batch_size=BATCH_SIZE)
    end = time.time()
    minutes, seconds = divmod(end-start, 60)
    print("{}\n".format('_' * 98))
    print("[{}] training completed ({}m {}s)".format(FILE, int(minutes), int(seconds)))

    # plot training
    plot_training(np.arange(EPOCHS), history.history)

    # save model
    save_model(args.model, model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-d', '--dataset', metavar='dataset', default='datasets/dynamics.npz')
    parser.add_argument('-m', '--model', metavar='model', default='models/dynamics.h5')
    parser.add_argument('-c', '--config', metavar='config', default='nonlinear')
    args = parser.parse_args()
    main(args)
