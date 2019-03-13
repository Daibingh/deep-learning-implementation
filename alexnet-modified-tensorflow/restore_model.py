# -*- coding: utf-8 -*-

"""restore model and params, and to make predictions"""

import tensorflow as tf
import pickle as pk
import numpy as np


def compute_test_acc():
    num_batch = 500
    num = x_val.shape[0]

    i = 0
    sum_acc = 0
    count = 0

    while i + num_batch <= num:
        x_val_batch = x_val[i:i + num_batch, :]
        y_val_batch = y_val[i:i + num_batch, :]
        x_val_batch = (x_val_batch - mean_v) / std_v

        acc, los, p = sess.run([accuracy, loss, pred], feed_dict={x: x_val_batch, y: y_val_batch})
        i += num_batch

        sum_acc += acc
        count += 1

    return sum_acc/count


if __name__ == '__main__':
    with open('data.pkl', 'rb') as f:
        data = pk.load(f)

    x_val = data.get('x_val')
    y_val = data.get('y_val')
    x_train = data.get('x_train')
    mean_v = np.mean(x_train)
    std_v = np.std(x_train)
    del data, x_train

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('./model/model.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./model/'))
        graph = sess.graph
        # get variable for input
        x = graph.get_tensor_by_name('x:0')
        y = graph.get_tensor_by_name('y:0')
        # get variable for prediction
        pred = graph.get_tensor_by_name('prediction:0')
        # get variable for accuracy
        accuracy = graph.get_tensor_by_name('accuracy:0')
        # get variable for loss
        loss = graph.get_tensor_by_name('loss:0')

        # num_batch = 500
        # N = x_val.shape[0]
        #
        # i = 0
        # sum_acc = 0
        # count = 0
        #
        # while i+num_batch <= N:
        #     x_val_batch = x_val[i:i+num_batch, :]
        #     y_val_batch = y_val[i:i+num_batch, :]
        #     x_val_batch = (x_val_batch-mean_v)/std_v
        #
        #     acc, los, p = sess.run([accuracy, loss, pred], feed_dict={x: x_val_batch, y: y_val_batch})
        #     i += num_batch
        #
        #     sum_acc += acc
        #     count += 1

        print('acc={}'.format(compute_test_acc()))


