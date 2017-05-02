# coding=utf-8
'''
Created on 2017年5月2日

@author: zjf
'''

'''
梯度下降（只支持单维度)
'''
from numpy import *

class gradientDescent(object):
    def __init__(self, alpha=0.01, epsilon=0.000001):
        self.alpha = alpha
        self.epsilon = epsilon  
        self.theta0 = 0
        self.theta1 = 0              
    
    def computeCostJ(self, x, y, theta0, theta1):
        m = len(y)
        s = 0
        for i in range(m):
            z = (theta0 + x[i] * theta1) - y[i]
            s += z ** 2
        return s / (2 * m)    
    
    def fit(self, x, y):
        m = len(y)   
        lastJ = 0
        times = 0
        while  True:
            sum0 = 0;sum1 = 0
            for i in range(m):
                d = y[i] - (self.theta0 + self.theta1 * x[i])
                sum0 += d
                sum1 += d * x[i]
            self.theta0 += self.alpha * sum0 / m
            self.theta1 += self.alpha * sum1 / m
            J = self.computeCostJ(x, y, self.theta0, self.theta1)
            if abs(J - lastJ) < self.epsilon:  # 比较上一次与这次的costFun差值，来判断是否结束
                break
            lastJ = J
            times += 1
        self.times = times
        self.intercept_ = self.theta0
        self.coef_ = self.theta1
        return self
        

'''
梯度下降（支持多维度)
'''
class gradientDescent2(object):
    def __init__(self, alpha=0.01, epsilon=0.000001):
        self.alpha = alpha
        self.epsilon = epsilon  
    
    def computeCostJ(self, X, Y, theta):
        m = len(Y)
        z = X * theta - Y  # X为m*n记录 theta为n*1 相乘后得m*1可与y相减
        z = multiply(z, z)  # 点乘，把m*1中的各元素计算平方
        return sum(z) / (2 * m)

    
    def fit(self, X, Y):
        m = len(Y)   
        lastJ = 0
        times = 0
        self.theta = zeros((shape(X)[1], 1))  # 定义为3*1的参数矩阵
        while  True:
            sum = (Y - X * self.theta).T * X
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
        
