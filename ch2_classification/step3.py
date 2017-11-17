# coding=utf-8
'''
Created on 2017年5月5日

@author: zjf
'''

'''
调用自己实现的Logisic回归
'''

from numpy import *
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression 
import matplotlib as mpl
import matplotlib.pyplot as plt


class MyLogisticRegression(object):
    def __init__(self, alpha=0.01, epsilon=0.00001):
        self.alpha = alpha
        self.epsilon = epsilon  
     
    def sigmod(self, inX):
        return 1.0 / (1 + exp(-inX))   
    
    def computeCostJ(self, X, Y, theta):
        theta = np.matrix(theta)
        X = np.matrix(X)
        y = np.matrix(Y)
        first = np.multiply(-y, np.log(self.sigmod(X * theta)))
        second = np.multiply((1 - y), np.log(1 - self.sigmod(X * theta)))
        return np.sum(first - second) / (len(X))
    
    def fit(self, X, Y):
        t = np.hstack((np.ones((X.shape[0], 1)), X))
        X = mat(t)
        Y = mat(Y)
        m = len(Y)   
        lastJ = 0
        times = 0
        self.theta = zeros((shape(X)[1], 1))  # 定义为3*1的参数矩阵
        while  True:
            sum = (Y - self.sigmod(X * self.theta)).T * X  # 与回归多了一个sigmod函数
            self.theta += self.alpha * sum.T / m
            J = self.computeCostJ(X, Y, self.theta)
            if abs(J - lastJ) < self.epsilon:  # 比较上一次与这次的costFun差值，来判断是否结束
                break
            lastJ = J
            times += 1
        self.times = times
        self.intercept_ = self.theta
        self.coef_ = self.theta
        return self

    def predict(self, X):
        t = np.hstack((np.ones((X.shape[0], 1)), X))
        x = mat(t)
        h = self.sigmod(x * self.theta)
        # Y = (-self.theta[0] - self.theta[1] * x) / self.theta[2]
        Y = array(h)
        return Y
    
if __name__ == '__main__':
#     data = pd.read_csv('testdata1.txt', header=None)
    data = pd.read_csv('ex2data1.txt', header=None)
    x, y = np.split(data.values, (2,), axis=1)
    
    
    mode = MyLogisticRegression()
    mode.fit(x, y)
    print mode.coef_, mode.intercept_
   
    # 用背景图色展示预测数据
    N, M = 500, 500  # x,y上切分多细
    t0 = np.linspace(x[:, 0].min(), x[:, 0].max(), N)
    t1 = np.linspace(x[:, 1].min(), x[:, 1].max(), N)
    x0, x1 = np.meshgrid(t0, t1)  # 填统到二维数组中
    
    x_test = np.stack((x0.flat, x1.flat), axis=1)
    y_hat = mode.predict(x_test).reshape(x0.shape)
    
    plt.pcolormesh(x0, x1, y_hat, cmap=mpl.colors.ListedColormap(['#77E0A0', '#FF8080']))
    
    
    # 展示原始数据
    plt.scatter(x[:, 0], x[:, 1], c=y[:, 0], cmap=mpl.colors.ListedColormap(['g', 'b']))
    
    plt.show()
    
    
    
