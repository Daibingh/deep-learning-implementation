# -*- coding: utf-8 -*-

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import utils


class Vgg16:

    def __init__(self, weights_file=None):
        self.weights_file = weights_file

    # def load_params(self):
    #     """load weights and biases from npy file"""
    #     pass

    def save_params(self, weights_file):
        """save weights and biases to npy file"""
        pass

    def build(self, input):
        start_time = time.time()
        if self.weights_file is None:  # random initial weights and biases
            self.wc1 = tf.Variable(tf.truncated_normal([3, 3, 3, 64], stddev=.01), name='wc1')
            self.wc2 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=.01), name='wc2')
            self.bc1 = tf.Variable(tf.constant(0.1, shape=[64]), name='bc1')
            self.bc2 = tf.Variable(tf.constant(0.0, shape=[64]), name='bc2')

            self.wc3 = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=.01), name='wc3')
            self.wc4 = tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev=.01), name='wc4')
            self.bc3 = tf.Variable(tf.constant(0.1, shape=[128]), name='bc3')
            self.bc4 = tf.Variable(tf.constant(0.0, shape=[128]), name='bc4')

            self.wc5 = tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev=.01), name='wc5')
            self.wc6 = tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev=.01), name='wc6')
            self.wc7 = tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev=.01), name='wc7')
            self.bc5 = tf.Variable(tf.constant(0.1, shape=[256]), name='bc5')
            self.bc6 = tf.Variable(tf.constant(0.0, shape=[256]), name='bc6')
            self.bc7 = tf.Variable(tf.constant(0.1, shape=[256]), name='bc7')

            self.wc8 = tf.Variable(tf.truncated_normal([3, 3, 256, 512], stddev=.01), name='wc8')
            self.wc9 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=.01), name='wc9')
            self.wc10 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=.01), name='wc10')
            self.bc8 = tf.Variable(tf.constant(0.1, shape=[512]), name='bc8')
            self.bc9 = tf.Variable(tf.constant(0.1, shape=[512]), name='bc9')
            self.bc10 = tf.Variable(tf.constant(0.1, shape=[512]), name='bc10')

            self.wc11 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=.01), name='wc11')
            self.wc12 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=.01), name='wc12')
            self.wc13 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=.01), name='wc13')
            self.bc11 = tf.Variable(tf.constant(0.1, shape=[512]), name='bc11')
            self.bc12 = tf.Variable(tf.constant(0.1, shape=[512]), name='bc12')
            self.bc13 = tf.Variable(tf.constant(0.1, shape=[512]), name='bc13')

            self.wf1 = tf.Variable(tf.truncated_normal([25088, 4096], stddev=.01), name='wf1')
            self.wf2 = tf.Variable(tf.truncated_normal([4096, 4096], stddev=.01), name='wf2')
            self.wf3 = tf.Variable(tf.truncated_normal([4096, 1000], stddev=.01), name='wf3')
            self.bf1 = tf.Variable(tf.constant(0.1, shape=[4096]), name='bf1')
            self.bf2 = tf.Variable(tf.constant(0.1, shape=[4096]), name='bf2')
            self.bf3 = tf.Variable(tf.constant(0.1, shape=[1000]), name='bf3')

        else:                          # load weights and biases from npy file
            data = np.load(self.weights_file)
            self.wc1 = tf.Variable(data['wc1'], name='wc1')
            self.wc2 = tf.Variable(data['wc2'], name='wc2')
            self.bc1 = tf.Variable(data['bc1'], name='bc1')
            self.bc2 = tf.Variable(data['bc2'], name='bc2')

            self.wc3 = tf.Variable(data['wc3'], name='wc3')
            self.wc4 = tf.Variable(data['wc4'], name='wc4')
            self.bc3 = tf.Variable(data['bc3'], name='bc3')
            self.bc4 = tf.Variable(data['bc4'], name='bc4')

            self.wc5 = tf.Variable(data['wc5'], name='wc5')
            self.wc6 = tf.Variable(data['wc6'], name='wc6')
            self.wc7 = tf.Variable(data['wc7'], name='wc7')
            self.bc5 = tf.Variable(data['bc5'], name='bc5')
            self.bc6 = tf.Variable(data['bc6'], name='bc6')
            self.bc7 = tf.Variable(data['bc7'], name='bc7')

            self.wc8 = tf.Variable(data['wc8'], name='wc8')
            self.wc9 = tf.Variable(data['wc9'], name='wc9')
            self.wc10 = tf.Variable(data['wc10'], name='wc10')
            self.bc8 = tf.Variable(data['bc8'], name='bc8')
            self.bc9 = tf.Variable(data['bc9'], name='bc9')
            self.bc10 = tf.Variable(data['bc10'], name='bc10')

            self.wc11 = tf.Variable(data['wc11'], name='wc11')
            self.wc12 = tf.Variable(data['wc12'], name='wc12')
            self.wc13 = tf.Variable(data['wc13'], name='wc13')
            self.bc11 = tf.Variable(data['bc11'], name='bc11')
            self.bc12 = tf.Variable(data['bc12'], name='bc12')
            self.bc13 = tf.Variable(data['bc13'], name='bc13')

            self.wf1 = tf.Variable(data['wf1'], name='wf1')
            self.wf2 = tf.Variable(data['wf2'], name='wf2')
            self.wf3 = tf.Variable(data['bc2'], name='wf3')
            self.bf1 = tf.Variable(data['bf1'], name='bf1')
            self.bf2 = tf.Variable(data['bf2'], name='bf2')
            self.bf3 = tf.Variable(data['bf3'], name='bf3')

        self.conv1 = tf.nn.relu(
            tf.nn.bias_add(tf.nn.conv2d(input, self.wc1, strides=[1, 1, 1, 1], padding='SAME'), self.bc1))
        self.conv2 = tf.nn.relu(
            tf.nn.bias_add(tf.nn.conv2d(self.conv1, self.wc2, strides=[1, 1, 1, 1], padding='SAME'), self.bc2))
        self.conv2 = tf.nn.max_pool(self.conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        self.conv3 = tf.nn.relu(
            tf.nn.bias_add(tf.nn.conv2d(self.conv2, self.wc3, strides=[1, 1, 1, 1], padding='SAME'), self.bc3))
        self.conv4 = tf.nn.relu(
            tf.nn.bias_add(tf.nn.conv2d(self.conv3, self.wc4, strides=[1, 1, 1, 1], padding='SAME'), self.bc4))
        self.conv4 = tf.nn.max_pool(self.conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        self.conv5 = tf.nn.relu(
            tf.nn.bias_add(tf.nn.conv2d(self.conv4, self.wc5, strides=[1, 1, 1, 1], padding='SAME'), self.bc5))
        self.conv6 = tf.nn.relu(
            tf.nn.bias_add(tf.nn.conv2d(self.conv5, self.wc6, strides=[1, 1, 1, 1], padding='SAME'), self.bc6))
        self.conv7 = tf.nn.relu(
            tf.nn.bias_add(tf.nn.conv2d(self.conv6, self.wc7, strides=[1, 1, 1, 1], padding='SAME'), self.bc7))
        self.conv7 = tf.nn.max_pool(self.conv7, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        self.conv8 = tf.nn.relu(
            tf.nn.bias_add(tf.nn.conv2d(self.conv7, self.wc8, strides=[1, 1, 1, 1], padding='SAME'), self.bc8))
        self.conv9 = tf.nn.relu(
            tf.nn.bias_add(tf.nn.conv2d(self.conv8, self.wc9, strides=[1, 1, 1, 1], padding='SAME'), self.bc9))
        self.conv10 = tf.nn.relu(
            tf.nn.bias_add(tf.nn.conv2d(self.conv9, self.wc10, strides=[1, 1, 1, 1], padding='SAME'), self.bc10))
        self.conv10 = tf.nn.max_pool(self.conv10, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        self.conv11 = tf.nn.relu(
            tf.nn.bias_add(tf.nn.conv2d(self.conv10, self.wc11, strides=[1, 1, 1, 1], padding='SAME'), self.bc11))
        self.conv12 = tf.nn.relu(
            tf.nn.bias_add(tf.nn.conv2d(self.conv11, self.wc12, strides=[1, 1, 1, 1], padding='SAME'), self.bc12))
        self.conv13 = tf.nn.relu(
            tf.nn.bias_add(tf.nn.conv2d(self.conv12, self.wc13, strides=[1, 1, 1, 1], padding='SAME'), self.bc13))
        self.conv13 = tf.nn.max_pool(self.conv13, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        self.fc1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(tf.reshape(self.conv13, [-1, 25088]), self.wf1), self.bf1))
        self.fc2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(self.fc1, self.wf2), self.bf2))
        self.fc3 = tf.nn.relu(tf.nn.bias_add(tf.matmul(self.fc2, self.wf3), self.bf3))

        self.prob = tf.nn.softmax(self.fc3, name="prob")

        print(("build model finished: %ds" % (time.time() - start_time)))


if __name__ == '__main__':
    # data = np.load('data/vgg16.npy', encoding='latin1')
    # print(type(data))
    # for k, v in data.item().items():
    #     print("{}: w shape={}; b shape={}".format(k, v[0].shape, v[1].shape))

    img1 = utils.load_image("./test_data/tiger.jpeg")
    img2 = utils.load_image('./test_data/puzzle.jpeg')
    img1_true_result = [1 if i == 292 else 0 for i in range(1000)]  # 1-hot result for tiger
    img2_true_result = [1 if i == 612 else 0 for i in range(1000)]

    batch1 = img1.reshape((1, 224, 224, 3))
    batch2 = img2.reshape((1, 224, 224, 3))
    true_out = tf.placeholder(tf.float32, [None, 1000])

    batch = np.concatenate((batch1, batch2), axis=0)
    true_result = np.array([img1_true_result, img2_true_result])

    vgg = Vgg16()
    x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
    vgg.build(x)
    cost = tf.reduce_sum((vgg.prob - true_out) ** 2)
    train = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        prob = sess.run(vgg.prob, feed_dict={x: batch})
        utils.print_prob(prob[0], './synset.txt')
        sess.run(train, feed_dict={x: batch, true_out: true_result})
        prob = sess.run(vgg.prob, feed_dict={x: batch})
        utils.print_prob(prob[0], './synset.txt')
