# -*- coding:utf-8 -*-
'''
Created on 2018年5月10日

@author: Jeffrey Zhou
'''

'''
自行实现神经网络
'''


import numpy as np
import matplotlib.pyplot as plt
import sklearn
import matplotlib as mpl
import tensorflow as tf
from sklearn.linear_model import LogisticRegression


def load_dataset():
    np.random.seed(1)  # 使得每次随机样本一致
    m = 400  # 样本数
    N = int(m/2)  # 每一类数量（共2类）
    D = 2  # 维度
    X = np.zeros((m, D))  # data matrix where each row is a single example
    # labels vector (0 for red, 1 for blue)
    Y = np.zeros((m, 1), dtype='uint8')
    a = 4  # maximum ray of the flower
    for j in range(2):
        ix = range(N*j, N*(j+1))
        t = np.linspace(j*3.12, (j+1)*3.12, N) + \
            np.random.randn(N)*0.2  # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.2  # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
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


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


if __name__ == '__main__':
    # 生成数据
    X, Y = load_dataset()
    print(np.shape(X), np.shape(Y))
    # showData(X,Y)

    # 初使化参数
    (n_x, m) = np.shape(X)
    n_y = 1  # y的类别
    tf.set_random_seed(1)
    parameters = {
        "W1": np.random.randn(4, n_x)*0.01,
        "b1": np.zeros((4, 1)),
        "W2": np.random.randn(n_y, 4)*0.01,
        "b2":  np.zeros((n_y, 1))
    }
    learning_rate = 1.2  # 学习率
    num_steps = 10000  # 总迭代次数

    def forward_propagation(X, params):
        '''
        向前传播，利用参数，从输入端计算到输出端
        '''
        w1 = params["W1"]
        b1 = params["b1"]
        w2 = params["W2"]
        b2 = params["b2"]
        z1 = np.dot(w1, X)+b1
        A1 = np.tanh(z1)
        z2 = np.dot(w2, A1)+b2
        y = sigmoid(z2)
        cache = {"Z1": z1, "A1": A1, "Z2": z2, "A2": y}
        return y, cache

    def costJ(y_hat, Y, params):
        '''
        根据预测函数，计算损失函数
        '''
        m = np.shape(Y)[1]
        costJ = -np.sum(Y*np.log(y_hat)+(1-Y)*np.log(1-y_hat))/m
        costJ = np.squeeze(costJ)
        return costJ

    def backward_propagation(parames, cache, X, Y):
        '''
        计算反射传播，根据导数以及损失值来反退出各层的梯度
        '''
        m = np.shape(Y)[1]
        dz2 = cache["A2"]-Y
        dw2 = np.dot(dz2, cache["A1"].T)/m
        db2 = np.sum(dz2, axis=1, keepdims=True)/m

        dz1 = np.dot(parames["W2"].T, dz2)*(1-np.power(cache["A1"], 2))
        dw1 = np.dot(dz1, X.T) / m
        db1 = np.sum(dz1, axis=1, keepdims=True)/m

        grads = {"dW1": dw1, "db1": db1, "dW2": dw2, "db2": db2}
        return grads

    # 循环梯度下降
    for i in range(num_steps):
        y_hat, cache = forward_propagation(X, parameters)
        cost = costJ(y_hat, Y, parameters)
        grads = backward_propagation(parameters, cache, X, Y)
        # 梯度更新
        parameters["W1"] = parameters["W1"] - learning_rate * grads["dW1"]
        parameters["b1"] = parameters["b1"] - learning_rate * grads["db1"]
        parameters["W2"] = parameters["W2"] - learning_rate * grads["dW2"]
        parameters["b2"] = parameters["b2"] - learning_rate * grads["db2"]
        if i % 1000 == 0:
            print("cost after iteration ", i, cost)

    def predict(X, params):
        '''
        预测函数
        '''
        a, _ = forward_propagation(X, params)
        return np.round(a)

    # 预测及展示
    y_hat = predict(X, parameters)
    print("ann accuracy:", np.mean(Y == y_hat)*100, "%")
    showData(X, Y, lambda testX: predict(testX.T, parameters))
