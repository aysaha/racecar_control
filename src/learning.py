#!/usr/bin/env python3

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from keras import models, layers, optimizers, utils

FILE = os.path.basename(__file__)
DIRECTORY = os.path.dirname(__file__)

K = 0                       # number of folds to process for validation
EPOCHS = 1000               # number of epochs to train the model
BATCH_SIZE = 1024           # training data batch size
SPLIT = 0.75                # split percentage for training vs. testing data

def load_dataset(path, training=False):
    print("[{}] loading dataset ({})".format(FILE, path))
    contents = np.load(path)
    data = contents['data']
    labels = contents['labels']

    if training is True:
        p = np.random.permutation(min(len(data), len(labels)))
        data, labels = data[p], labels[p]
    else:
        data = list(data)
        labels = list(labels)

    return data, labels

def save_dataset(path, data, labels, array=False):
    print("[{}] saving dataset ({})".format(FILE, path))
    data = np.array(data)
    labels = np.array(labels)
    np.savez_compressed(path, data=data, labels=labels)

def build_model(inputs, outputs, summary=False):
    model = models.Sequential()

    model.add(layers.Dense(512, activation='relu', input_shape=(inputs,)))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(outputs, activation='linear'))
    model.compile(loss='mse', optimizer='rmsprop', metrics=['acc'])

    if summary is True:
        model.summary()

    return model

def k_fold_cross_validation(data, labels, k, epochs, batch_size):
    result = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}
    samples = data.shape[0] // k

    for i in range(k):
        print("Processing fold {}/{}".format(i+1, k))

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
                            epochs=epochs,
                            batch_size=batch_size)

        # record scores
        result['loss'].append(history.history['loss'])
        result['acc'].append(history.history['acc'])
        result['val_loss'].append(history.history['val_loss'])
        result['val_acc'].append(history.history['val_acc'])

        print("")

    # average results
    result['loss'] = np.mean(result['loss'], axis=0)
    result['acc'] = np.mean(result['acc'], axis=0)
    result['val_loss'] = np.mean(result['val_loss'], axis=0)
    result['val_acc'] = np.mean(result['val_acc'], axis=0)

    return result

def visualize_training(epochs, result):
        plt.subplots(2)

        # loss
        plt.subplot(2, 1, 1)  
        plt.plot(epochs, result['loss'], '--b', label="Training")
        plt.plot(epochs, result['val_loss'], '-g', label="Validation")
        plt.title('Model Performance')
        plt.ylabel('Loss')
        plt.grid()
        plt.legend()

        # accuracy
        plt.subplot(2, 1, 2)  
        plt.plot(epochs, result['acc'], '--b', label="Training")
        plt.plot(epochs, result['val_acc'], '-g', label="Validation")
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid()
        plt.legend()

        plt.show()

def main(args):
    assert args.file is not None and os.path.splitext(args.file)[1] == '.npz'

    # load dataset
    data, labels = load_dataset(args.file, training=True)

    if K > 0:
        # format dataset
        samples = data.shape[0]
        index = int(samples * SPLIT)
        train_data = data[:index]
        train_labels = labels[:index]
        test_data = data[index:]
        test_labels = labels[index:]

        # perform K-fold cross-validation
        result = k_fold_cross_validation(train_data, train_labels, K, EPOCHS, BATCH_SIZE)

        # visualize training
        visualize_training(np.arange(EPOCHS), result)
    else:
        # build model
        model = build_model(data.shape[1], labels.shape[1], summary=True)

        # train model
        model.fit(data, labels, epochs=EPOCHS, batch_size=BATCH_SIZE)

        # save model
        model.save('models/dynamics.h5')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='system dynamics dataset')
    args = parser.parse_args()
    main(args)