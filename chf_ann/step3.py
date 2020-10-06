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
import sklearn.datasets
import matplotlib as mpl
import tensorflow as tf


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
    train_X, train_Y, test_X, test_Y = load_dataset()
    print(np.shape(train_X), np.shape(train_Y))
#     showData(train_X, train_Y)
    train_X = train_X.T
    train_Y = train_Y.T
    
    tf.random.set_seed(0)
    
    learning_rate = 0.001
    num_steps = 10000  # 总迭代次数
    
    #zero初使化
#     model = tf.keras.models.Sequential([
#         tf.keras.layers.Dense(units=4,
#                               kernel_initializer=tf.keras.initializers.Zeros(),
#                               bias_initializer=tf.keras.initializers.Zeros(),
#                               activation=tf.nn.tanh),
#         tf.keras.layers.Dense(units=1,
#                               kernel_initializer=tf.keras.initializers.Zeros(),
#                               bias_initializer=tf.keras.initializers.Zeros(),
#                               activation=tf.nn.sigmoid)])
    #随机初使化
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=4,
                              kernel_initializer=tf.keras.initializers.RandomNormal(),
                              bias_initializer=tf.zeros_initializer(),
                              activation=tf.nn.tanh),
        tf.keras.layers.Dense(units=1,
                              kernel_initializer=tf.keras.initializers.RandomNormal(),
                              bias_initializer=tf.zeros_initializer(),
                              activation=tf.nn.sigmoid)])
    
    # 定义反向传播的优化器
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
     
    costs =[]
    # 模型训练
    for i in range(num_steps):
        with tf.GradientTape() as tape:
            y_pred = model(train_X)
            # 定义损失函数
            loss = tf.losses.mean_squared_error(train_Y, y_pred)
        if i % 1000 == 0:
            print("batch %d: loss %f" % (i, tf.reduce_mean(loss).numpy()))
            costs.append(tf.reduce_mean(loss).numpy())
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
 
    # 预测，并输出
    y_hat = tf.round(model(test_X.T)).numpy()
    print("ann accuracy:", np.mean(test_Y.T == y_hat) * 100, "%")
    showData(train_X.T, train_Y.T, lambda testX: tf.round(model(testX)).numpy())
    
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
