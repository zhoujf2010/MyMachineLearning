# -*- coding:utf-8 -*-
'''
Created on 2018年5月9日

@author: Jeffrey Zhou
'''

'''
使用tensorflow实现一个简单的神经网络
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
from sklearn.linear_model import LogisticRegression


def load_dataset():
    np.random.seed(1)  # 使得每次随机样本一致
    m = 400  # 样本数
    N = int(m / 2)  # 每一类数量（共2类）
    D = 2  # 维度
    X = np.zeros((m, D))  # data matrix where each row is a single example
    # labels vector (0 for red, 1 for blue)
    Y = np.zeros((m, 1), dtype='uint8')
    a = 4  # maximum ray of the flower
    for j in range(2):
        ix = range(N * j, N * (j + 1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + \
            np.random.randn(N) * 0.2  # theta
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2  # radius
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j

    X = X.T
    Y = Y.T
    return X, Y


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
    X, Y = load_dataset()
    print(np.shape(X), np.shape(Y))
    X = X.T
    Y = Y.T
    # showData(X,Y)

    # 采用Logistic回归进行预测
#     mode = LogisticRegression()
#     mode.fit(X.T, Y.T)
#     y_hat = mode.predict(X.T)    
#     y_hat = y_hat.reshape((-1, 1))
#     print("logistic accuracy:", np.mean(Y == y_hat.T)*100, "%")
    # showData(X, Y, lambda testX: mode.predict(testX))
    
    tf.random.set_seed(0)
    
    learning_rate = 0.001
    num_steps = 10000  # 总迭代次数
    
    #定义模型步骤
    class MyModel(tf.keras.Model):

        def __init__(self):
            super().__init__()
            self.L1 = tf.keras.layers.Dense(units=4, activation=tf.nn.tanh)
            self.L2 = tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid)
        
        def call(self, inputs):
            x = self.L1(inputs)
            x = self.L2(x)
            return x
        
    
    model = MyModel()
    # 定义反向传播的优化器
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    
    # 模型训练
    for i in range(num_steps):
        with tf.GradientTape() as tape:
            y_pred = model(X)
            # 定义损失函数
            loss = tf.reduce_mean(tf.square(y_pred - Y))
        if i % 1000 == 0:
            print("batch %d: loss %f" % (i, loss.numpy()))
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

#     tf.summary.FileWriter("log", sess.graph)

    # 预测，并输出
    y_hat = tf.round(model(X)).numpy()
    print("ann accuracy:", np.mean(Y == y_hat) * 100, "%")
    showData(X.T, Y.T, lambda testX: tf.round(model(testX)).numpy())
