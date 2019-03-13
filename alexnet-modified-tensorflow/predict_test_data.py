# -*- coding: utf-8 -*-

import pickle as pk
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd


def predict_by_batch(xx, batch_size):
    num = xx.shape[0]
    pred = np.zeros((num))
    i = 0

    while i+batch_size <= num:
        p = sess.run([prediction], {x: xx[i:i+batch_size, :]})
        pred[i:i+batch_size] = p[0]
        i += batch_size
    return pred


def show_img(index, pred, img):
    fig = plt.figure('show result')
    ax = fig.add_subplot(111)
    ax.imshow(img[index, :].reshape(28, 28), cmap='gray')
    ax.set_title('predict result: {}, index: {}'.format(pred[index], index))
    plt.show()


if __name__ == '__main__':
    with open('data.pkl', 'rb') as f:
        data = pk.load(f)

    # print(data.keys())
    x_test = data.get('x_test')
    x_train = data.get('x_train')
    mean_v = np.mean(x_train)
    std_v = np.std(x_train)
    del data, x_train
    # print(mean_v, std_v)
    N = x_test.shape[0]
    # print(N)

    # x_test = (x_test-mean_v)/std_v
    #
    # with tf.Session() as sess:
    #     saver = tf.train.import_meta_graph('./model/model.meta')
    #     saver.restore(sess, tf.train.latest_checkpoint('./model/'))
    #     graph = sess.graph
    #     # get variable for input
    #     x = graph.get_tensor_by_name('x:0')
    #     # get variable for prediction
    #     prediction = graph.get_tensor_by_name('prediction:0')
    #     # pred = sess.run([prediction], feed_dict={x: x_test})
    #     pred = predict_by_batch(x_test, 800)
    #     print(len(pred), x_test.shape[0])
    #     print(pred)
    #
    #     with open('prediction.pkl', 'wb') as f:
    #         pk.dump(pred, f)

    with open('prediction.pkl', 'rb') as f:
        pred = pk.load(f)

    show_img(np.random.randint(28000), pred, x_test)

    # results = pd.Series(pred.astype(int), name="Label")
    # submission = pd.concat([pd.Series(range(1, 28001), name="ImageId"), results], axis=1)
    # submission.to_csv("mnist_prediction.csv", index=False)