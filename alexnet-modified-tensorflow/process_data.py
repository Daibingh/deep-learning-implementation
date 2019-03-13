# -*- coding: utf-8 -*-

import pandas as pd
import pickle as pk
import numpy as np


if __name__ == '__main__':
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    # print(train.head())
    # print(test.head())
    y_train = train['label']
    # print(type(y_train), y_train)
    x_train = train.drop(labels=['label'], axis=1)

    # check for missing values
    # print(train.isnull().any())
    # print(test.isnull().any())
    del train

    x_train = x_train.values
    y_train = y_train.values

    # print(type(x_train), type(y_train), x_train.shape, y_train.shape)
    x_test = test.values
    # print(type(x_test), x_test.shape)

    n_class = 10
    temp = np.zeros((y_train.shape[0], n_class))
    temp[range(0, y_train.shape[0]), y_train] = 1  # convert to one hot code.
    y_train = temp
    del temp

    # divide val dataset
    N = x_train.shape[0]
    ratio_of_val = .2
    num_val = int(N*ratio_of_val)
    index = np.arange(N)
    np.random.shuffle(index)
    x_val = x_train[index[:num_val], :]
    y_val = y_train[index[:num_val], :]
    x_train = x_train[index[num_val:], :]
    y_train = y_train[index[num_val:], :]

    data = {
        'x_train': x_train,
        'y_train': y_train,
        'x_val': x_val,
        'y_val': y_val,
        'x_test': x_test
    }
    with open('data.pkl', 'wb') as f:
        pk.dump(data, f)

