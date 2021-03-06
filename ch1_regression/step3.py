# coding=utf-8
'''
Created on 2017年5月2日

@author: zjf
'''

'''
自己实现批梯度下降
'''

from numpy import *
import pandas as pd;
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np;
from sklearn.linear_model import LinearRegression

def not_empty(s):
    return s != ''

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
    
    def predict(self,x):
        y = self.theta0 + x * self.theta1
        return y
    
if __name__ == '__main__':
    np.set_printoptions(suppress=True)  # 控制print时，不要用科学计数法显示
    datard = pd.read_csv("housing.data", header=None)
    data = np.empty((len(datard), 14))  # 创建一个N行14列的数据
    for i, row in enumerate(datard.values):
        data[i] = map(float, filter(not_empty, row[0].split(' ')))  # 处理一行数据，拆分到数组中
    
#     print "data=\n",data
    x = data[:, 7:8]
    y = data[:, 13:14]
    
    
    mode = gradientDescent(alpha = 0.01,epsilon = 0.000001)
    mode.fit(x, y)
    print mode.coef_, mode.intercept_
    
    y_hat = mode.predict(x)  # 用模型直接预测数据
     
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.plot(x, y, 'rx', label=u'原始数据')
    plt.plot(x, y_hat, 'g-', label=u'预测数据')
    plt.legend(loc="upper right")
    plt.show()
