# coding=utf-8
'''
Created on 2017年5月5日

@author: zjf
'''

from numpy import *
import numpy as np


class MyLogisticRegression(object):
    def __init__(self, alpha=0.01, epsilon=0.00001):
        self.alpha = alpha
        self.epsilon = epsilon  
        self.theta0 = 0
        self.theta1 = 0   
     
    def sigmod(self, inX):
        return 1.0 / (1 + exp(-inX))   
    
    def computeCostJ(self,X, Y,theta):
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
            sum = (Y - self.sigmod(X * self.theta)).T * X   #与回归多了一个sigmod函数
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
