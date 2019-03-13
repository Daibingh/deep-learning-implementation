# -*- coding: utf-8 -*-


from alexnet import alex_net
from alexnet import weights, biases, x, y
import pickle as pk
import tensorflow as tf
import numpy as np
import shutil as stl
import os.path
import matplotlib.pyplot as plt

# load data
with open('data.pkl', 'rb') as f:
    data = pk.load(f)

# set some vars
model_dir = './model'
logdir = './logdir/run1'
if os.path.exists(logdir):
    stl.rmtree(logdir)

# hyper params
lr = .01
batch_size = 400

# training and validating data
x_train = data.get('x_train')
y_train = data.get('y_train')
x_val = data.get('x_val')
y_val = data.get('y_val')
x_test = data.get('x_test')
N = x_train.shape[0]
mean_v = np.mean(x_train)
std_v = np.std(x_train)
x_train = (x_train - mean_v) / std_v
x_val = (x_val-mean_v) / std_v

epochs = 100
num_batch_one_epoch = N // batch_size
num_iter = epochs * num_batch_one_epoch
del data


# def compute_test_acc_los(x, y, batch_size):
#     sum_acc = 0
#     sum_los = 0
#     n = x.shape[0]
#     count = 0
#     i = 0
#
#     while i+batch_size <= n:
#         xx = x[i:i+batch_size, :]
#         yy = y[i:i+batch_size, :]
#         yy_pre = alex_net(xx, weights, biases)
#         acc = compute_acc(yy_pre, yy)
#         sum_acc += acc
#
#         los = - tf.reduce_mean(tf.reduce_sum(yy*tf.log(yy_pre), axis=1), axis=0, name='loss')
#         sum_los += los
#
#         i += batch_size
#         count += 1
#
#     acc = sum_acc / count
#     los = sum_los / count
#     return acc, los


# def compute_acc(y_pre, y):
#     max_index1 = tf.argmax(y_pre, axis=1)
#     max_index2 = tf.argmax(y, axis=1)
#     equaled = tf.equal(max_index1, max_index2)
#     return tf.reduce_mean(tf.cast(equaled, tf.float32))


def get_next_batch(x_data, y_data, num):
    """get one batch of x, y randomly"""
    index_all = np.arange(x_data.shape[0])
    np.random.shuffle(index_all)
    index_selected = index_all[:num]
    return x_data[index_selected, :], y_data[index_selected, :]


def variable_summaries(tag, var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    mean = tf.reduce_mean(var)
    tf.summary.scalar(tag+'/mean', mean)
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar(tag+'/stddev', stddev)
    tf.summary.scalar(tag+'/max', tf.reduce_max(var))
    tf.summary.scalar(tag+'/min', tf.reduce_min(var))
    tf.summary.histogram(tag+'/histogram', var)


if __name__ == '__main__':

    # forward
    y_pre = alex_net(x, weights, biases)

    # compute loss
    loss = - tf.reduce_mean(tf.reduce_sum(y*tf.log(y_pre), axis=1), axis=0, name='loss')

    # optimize
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
    # train_op = optimizer.minimize(loss)

    # gradients clipping
    gvs = optimizer.compute_gradients(loss)
    capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
    train_op = optimizer.apply_gradients(capped_gvs)

    # compute accuracy for train data (small data)
    max_index1 = tf.argmax(y_pre, axis=1, name='prediction')
    max_index2 = tf.argmax(y, axis=1)
    equaled = tf.equal(max_index1, max_index2)
    accuracy = tf.reduce_mean(tf.cast(equaled, tf.float32), name='accuracy')

    # add summaries
    with tf.name_scope('summary'):
        accuracy_summary = tf.summary.scalar('accuracy', accuracy)
        loss_summary = tf.summary.scalar('loss', loss)
        # summary for weights
        variable_summaries('wc1', weights['wc1'])
        variable_summaries('wc2', weights['wc2'])
        variable_summaries('wc3', weights['wc3'])
        variable_summaries('wc4', weights['wc4'])
        variable_summaries('wc5', weights['wc5'])
        variable_summaries('wf1', weights['wf1'])
        variable_summaries('wf2', weights['wf2'])
        variable_summaries('wf3', weights['wf3'])

        # summary for gradients
        for grad, var in gvs:
            variable_summaries('grads/'+var.name, grad)

    merged = tf.summary.merge_all()

    # start session and computation
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(logdir+'/train', sess.graph)
        test_writer = tf.summary.FileWriter(logdir+'/test')
        saver = tf.train.Saver(max_to_keep=5)

        init = tf.global_variables_initializer()
        sess.run(init)

        # save model alone
        saver.save(sess, model_dir+'/model')
        # os._exit(0)

        # fig1 = plt.figure('accuracy and loss')
        # ax11 = fig1.add_subplot(121)
        # ax12 = fig1.add_subplot(122)
        #
        # acc_train = []
        # acc_val = []
        # los_train = []
        # los_val = []
        # n_iter = []

        for i in range(num_iter):
            x_batch, y_batch = get_next_batch(x_train, y_train, batch_size)
            sess.run(train_op, feed_dict={x: x_batch, y: y_batch})

            if i % 10 == 0:
                los1, acc1, mg = sess.run([loss, accuracy, merged], feed_dict={x: x_batch, y: y_batch})

                sum_los = 0
                sum_acc = 0
                j = 0
                count = 0

                while j+batch_size <= x_val.shape[0]:
                    los2, acc2 = sess.run([loss, accuracy], feed_dict={x: x_val[j:j+batch_size, :], y: y_val[j:j+batch_size, :]})
                    j += batch_size
                    count += 1
                    sum_los += los2
                    sum_acc += acc2

                los2 = sum_los / count
                acc2 = sum_acc / count

                train_writer.add_summary(mg, i)
                # test_writer.add_summary(los_smy, i)
                # test_writer.add_summary(acc_smy, i)

                # n_iter.append(i)
                # acc_train.append(acc1)
                # acc_val.append(acc2)
                # los_train.append(los1)
                # los_val.append(los2)

                print("epoch: {}, iter: {}, train_batch: accuracy = {:.4f}, loss = {:.2f}, "
                      "val_dataset: accuracy = {:.4f}, loss = {:.2f}".format(i // num_batch_one_epoch, i, acc1, los1, acc2, los2))

                model_name = '{}/model-{}-{}'.format(model_dir, int(acc1*1000), int(acc2*1000))

                if acc1 > .97 and acc2 > .965:
                    saver.save(sess, model_name, i, write_meta_graph=False)  # save model and params

                # ax11.plot(n_iter, acc_train, 'b')
                # ax11.plot(n_iter, acc_val, 'r')
                # ax12.plot(n_iter, los_train, 'b')
                # ax12.plot(n_iter, los_val, 'r')
                #
                # plt.draw()
                # plt.pause(0.001)
                #
                # ax11.cla()
                # ax12.cla()
