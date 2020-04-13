#!/usr/bin/env python

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from keras import models, layers, optimizers, utils

FILE = os.path.basename(__file__)
DIRECTORY = os.path.dirname(__file__)

EPOCHS = 100
BATCH_SIZE = 32

def save_dataset(path, data, labels):
    print("[{}] saving dataset ({})".format(FILE, path))
    data, labels = np.array(data), np.array(labels)
    np.savez_compressed(path, data=data, labels=labels)

def load_dataset(path, shuffle=False):
    print("[{}] loading dataset ({})".format(FILE, path))
    contents = np.load(path)
    data, labels = contents['data'], contents['labels']

    if shuffle is True:
        p = np.random.permutation(min(len(data), len(labels)))
        data, labels = data[p], labels[p]

    data, labels = list(data), list(labels)
    return data, labels

def save_model(path, model):
    print("[{}] saving model ({})".format(FILE, path))
    model.save(path)

def load_model(path):
    print("[{}] loading model ({})".format(FILE, path))
    model = models.load_model(path)
    return model

def build_model(inputs, outputs, summary=False):
    model = models.Sequential()

    model.add(layers.Dense(16, activation='relu', input_shape=(inputs,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(outputs, activation='linear'))
    model.compile(loss='mse', optimizer='rmsprop', metrics=['acc'])

    if summary is True:
        model.summary()

    return model

def k_fold_cross_validation(data, labels, epochs, batch_size, K=4):
    results = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}
    samples = data.shape[0] // K

    for i in range(K):
        print("[{}] processing fold ({}/{})".format(FILE, i+1, K))

        # validation data and lables
        val_data = data[samples*i:samples*(i+1)]
        val_labels = labels[samples*i:samples*(i+1)]

        # training data and labels
        train_data = np.concatenate([data[:samples*i], data[samples*(i+1):]], axis=0)
        train_labels = np.concatenate([labels[:samples*i], labels[samples*(i+1):]], axis=0)

        # build model
        model = build_model(data.shape[1], labels.shape[1])

        # train model
        history = model.fit(train_data, train_labels,
                            validation_data=(val_data, val_labels),
                            epochs=epochs, batch_size=batch_size)

        # record scores
        results['loss'].append(history.history['loss'])
        results['acc'].append(history.history['acc'])
        results['val_loss'].append(history.history['val_loss'])
        results['val_acc'].append(history.history['val_acc'])
        print("")

    # average results
    results['loss'] = np.mean(results['loss'], axis=0)
    results['acc'] = np.mean(results['acc'], axis=0)
    results['val_loss'] = np.mean(results['val_loss'], axis=0)
    results['val_acc'] = np.mean(results['val_acc'], axis=0)

    return results

def plot_training(epochs, results):
    print("[{}] plotting training results".format(FILE))
    plt.subplots(2)

    # loss
    plt.subplot(2, 1, 1)  
    plt.plot(epochs, results['loss'], '--b', label="Training")
    plt.plot(epochs, results['val_loss'], '-g', label="Validation")
    plt.title('Model Performance')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()

    # accuracy
    plt.subplot(2, 1, 2)  
    plt.plot(epochs, results['acc'], '--b', label="Training")
    plt.plot(epochs, results['val_acc'], '-g', label="Validation")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.legend()

    plt.show()

def main(args):
    assert os.path.exists(args.dataset) and os.path.splitext(args.dataset)[1] == '.npz'
    assert args.model is None or os.path.splitext(args.model)[1] == '.h5'

    # format dataset
    data, labels = load_dataset(args.dataset, shuffle=True)
    data, labels = np.array(data), np.array(labels)

    if args.model is None:
        # perform K-fold cross-validation
        results = k_fold_cross_validation(data, labels, EPOCHS, BATCH_SIZE)

        # plot training
        plot_training(np.arange(EPOCHS), results)
    else:
        # build model
        model = build_model(data.shape[1], labels.shape[1], summary=True)

        # train model
        model.fit(data, labels, epochs=EPOCHS, batch_size=BATCH_SIZE)

        # save model
        save_model(args.model, model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-d', '--dataset', metavar='dataset', default='datasets/dynamics.npz')
    parser.add_argument('-m', '--model', metavar='model')
    args = parser.parse_args()
    main(args)