#!/usr/bin/env python

import os
import time
import argparse

import numpy as np
import matplotlib.pyplot as plt
from keras import models, layers

FILE = os.path.basename(__file__)
DIRECTORY = os.path.dirname(__file__)

EPOCHS = 10
BATCH_SIZE = 32

def dynamics(q, u, model):
    assert q.shape == (6,) or q.shape == (6, 1)
    assert u.shape == (3,) or q.shape == (3, 1)
    assert model is not None

    q = q.reshape(1, -1)
    u = u.reshape(1, -1)
    F = model.predict([q, u])[0]

    return F

def horizon(q, u, model, steps):
    assert q.shape == (6,) or q.shape == (6, 1)
    assert u.shape == (steps, 3)
    assert model is not None
    assert steps > 0

    q = np.vstack((q, np.zeros((steps, q.shape[0]))))

    for i in range(steps):
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

def build_model(form, N, M):
    print("[{}] building model ({})".format(FILE, form))

    # define input layers
    q = layers.Input(shape=(N,), name='q')
    u = layers.Input(shape=(M,), name='u')
    q_u = layers.Concatenate(name ='q_u')([q, u])

    if form == 'nonlinear':
        # define hidden layers
        hidden_layer = layers.Dense(16, activation='relu', name='hidden_layer')(q_u)
        output_layer = layers.Dense(N, activation='linear', name='output_layer')(hidden_layer)
        
        # define output layer
        F = layers.Reshape((N,), name='F')(output_layer)
    elif form == 'affine':
        # define hidden layers for f
        hidden_layer_f = layers.Dense(8, activation='relu', name='hidden_layer_f')(q)
        output_layer_f = layers.Dense(N, activation='linear', name='output_layer_f')(hidden_layer_f)
        f = layers.Reshape((N,), name='f')(output_layer_f)

        # define hidden layers for g
        hidden_layer_g = layers.Dense(8, activation='relu', name='hidden_layer_g')(q)
        output_layer_g = layers.Dense(N*M, activation='linear', name='output_layer_g')(hidden_layer_g)
        g = layers.Reshape((N, M), name='g')(output_layer_g)
        gu = layers.Dot(-1, name='gu')([g, u])  

        # define output layer
        F = layers.Add(name='F')([f, gu])
    elif form == 'linear':
        # define hidden layers for A
        hidden_layer_A = layers.Dense(8, activation='relu', name='hidden_layer_A')(q_u)
        output_layer_A = layers.Dense(N*N, activation='linear', name='output_layer_A')(hidden_layer_A)
        A = layers.Reshape((N, N), name='A')(output_layer_A)
        Aq = layers.Dot(-1, name='Aq')([A, q])

        # define hidden layers for B
        hidden_layer_B = layers.Dense(8, activation='relu', name='hidden_layer_B')(q_u)
        output_layer_B = layers.Dense(N*M, activation='linear', name='output_layer_B')(hidden_layer_B)
        B = layers.Reshape((N, M), name='B')(output_layer_B)
        Bu = layers.Dot(-1, name='Bu')([B, u])

        # define output layer
        F = layers.Add(name='F')([Aq, Bu])
    
    # create model
    model = models.Model(inputs=[q, u], outputs=F, name=form)
    model.compile(loss='mse', optimizer='rmsprop', metrics=['acc'])

    return model

def train_model(model, q, u, F):
    print("[{}] training started".format(FILE))
    print("{}".format('_' * 98))
    model.summary()

    # record training progress
    start = time.time()
    history = model.fit([q, u], F, epochs=EPOCHS, batch_size=BATCH_SIZE)
    end = time.time()

    minutes, seconds = divmod(end-start, 60)
    print("{}\n".format('_' * 98))
    print("[{}] training completed ({}m {}s)".format(FILE, int(minutes), int(seconds)))
    
    return history.history

def plot_training(results):
    print("[{}] plotting training results".format(FILE))
    plt.subplots(2, num='training_results')
    epochs = np.arange(len(results['loss']))

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
    assert args.model in ['nonlinear', 'affine', 'linear']

    # load dataset
    q, u, F = load_dataset(args.dataset, shuffle=True)
    N, M = q.shape[1], u.shape[1]

    # build model
    model = build_model(args.model, N, M)

    # train model
    results = train_model(model, q, u, F)

    # plot training
    #plot_training(results)

    # save model
    save_model('models/dynamics_{}.h5'.format(args.model), model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-d', '--dataset', metavar='dataset', default='datasets/dynamics.npz')
    parser.add_argument('-m', '--model', metavar='model', default='nonlinear')
    args = parser.parse_args()
    main(args)
