# -*- coding:utf-8 -*-
'''
Created on 2018年5月10日

@author: Jeffrey Zhou
'''

'''
验证初使化参数对结果影响
'''

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import matplotlib as mpl
import tensorflow as tf
from sklearn.linear_model import LogisticRegression


def load_dataset():
    np.random.seed(1)
    train_X, train_Y = sklearn.datasets.make_circles(n_samples=300, noise=.05)
    np.random.seed(2)
    test_X, test_Y = sklearn.datasets.make_circles(n_samples=100, noise=.05)
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    test_X = test_X.T
    test_Y = test_Y.reshape((1, test_Y.shape[0]))
    return train_X, train_Y, test_X, test_Y


def showData(X, Y, predict=None):
    # 显示标注数据
    N, M = 500, 500  # x,y上切分多细
    t0 = np.linspace(X[0, :].min(), X[0, :].max(), N)
    t1 = np.linspace(X[1, :].min(), X[1, :].max(), N)
    x0, x1 = np.meshgrid(t0, t1)  # 填统到二维数组中
    x_test = np.stack((x0.flat, x1.flat), axis=1)

    if predict != None:
        y_hat = predict(x_test).reshape(x0.shape)
        plt.pcolormesh(x0, x1, y_hat, cmap=mpl.colors.ListedColormap(
            ['#77E0A0', '#FF8080']))

    # 显示原数据
    plt.scatter(X[0, :], X[1, :], c=Y[0], s=40, cmap=plt.cm.Spectral)
    plt.show()


if __name__ == '__main__':
    # 生成数据
    # load image dataset: blue/red dots in circles
    train_X, train_Y, test_X, test_Y = load_dataset()
    print(np.shape(train_X), np.shape(train_Y))
    # showData(train_X, train_Y)
    # 定义占位符(placeholder)
    (n_x, m) = np.shape(train_X)
    n_y = 1  # y的类别
    X_hold = tf.placeholder("float", shape=[n_x, None], name="X")
    Y_hold = tf.placeholder("float", shape=[n_y, None], name="Y")

    # 初使化参数 normal
    tf.set_random_seed(1)
    parameters = {
        "W1": tf.Variable(tf.random_normal([4, n_x])),
        "b1": tf.Variable(tf.zeros([4, 1])),
        "W2": tf.Variable(tf.random_normal([n_y, 4])),
        "b2": tf.Variable(tf.zeros([n_y, 1]))
    }
    # 初使化参数 zero
#     tf.set_random_seed(1)
#     parameters = {
#         "W1": tf.Variable(tf.zeros([4, n_x])),
#         "b1": tf.Variable(tf.zeros([4, 1])),
#         "W2": tf.Variable(tf.zeros([n_y, 4])),
#         "b2": tf.Variable(tf.zeros([n_y, 1]))
#     }
    learning_rate = 1.2  # 学习率
    num_steps = 10000  # 总迭代次数
 
    # 定义向前传播
    Z1 = tf.add(tf.matmul(parameters['W1'], X_hold), parameters['b1'])
    A1 = tf.nn.sigmoid(Z1)
    Z2 = tf.add(tf.matmul(parameters['W2'], A1), parameters['b2'])
    A2 = tf.nn.sigmoid(Z2)
    Y_Out = tf.round(A2)
 
    # 定义损失函数
    costJ = tf.reduce_mean(-Y_hold * tf.log(A2) - (1 - Y_hold) * tf.log(1 - A2))
 
    # 定义反向传播的优化器
    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=learning_rate).minimize(costJ)
 
    # 模型训练
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(num_steps):
        _, cost = sess.run([optimizer, costJ], feed_dict={
            X_hold: train_X, Y_hold: train_Y})
        if i % 1000 == 0:
            print("Cost after epoch %i: %f" % (i, cost))
 
    # 写入可视化
    #tf.summary.FileWriter("log", sess.graph)
 
    # 预测，并输出
    y_hat = sess.run(Y_Out, feed_dict={X_hold: train_X})
    print("ann accuracy:", np.mean(train_Y == y_hat) * 100, "%")
    showData(train_X, train_Y, lambda testX: sess.run(
        Y_Out, feed_dict={X_hold: testX.T}))
 
    sess.close()  # 关闭对象
