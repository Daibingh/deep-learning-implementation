# -*- coding: utf-8 -*-

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


class Vgg16:

    def __init__(self, weights_file=None):
        self.weights_file = weights_file

    def load_params(self):
        """load weights and biases from npy file"""
        pass

    def save_params(self, weights_file):
        """save weights and biases to npy file"""
        pass

    def build(self, input):
        if self.weights_file is None:  # random initial weights and biases
            pass
            # self.wc1 = tf.Variable(tf.truncated_normal([ ], stddev= ), name='wc1')
            # self.wc2 = tf.Variable(tf.truncated_normal([ ], stddev= ), name='wc2')
            # self.wc3 = tf.Variable(tf.truncated_normal([ ], stddev= ), name='wc3')
            # self.wc4 = tf.Variable(tf.truncated_normal([ ], stddev= ), name='wc4')
        else:                          # load weights and biases from npy file
            pass


if __name__ == '__main__':
    data = np.load('data/vgg16.npy', encoding='latin1')
    print(type(data))
    for k, v in data.item().items():
        print("{}: w shape={}; b shape={}".format(k, v[0].shape, v[1].shape))
