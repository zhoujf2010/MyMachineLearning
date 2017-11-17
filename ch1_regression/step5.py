# coding=utf-8
'''
Created on 2017年5月2日

@author: zjf
'''

'''
自己实现随机机梯度下降
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
    
    def computeCostJ(self, X, Y, theta):
        m = len(Y)
        z = X * theta - Y  # X为m*n记录 theta为n*1 相乘后得m*1可与y相减
        z = multiply(z, z)  # 点乘，把m*1中的各元素计算平方
        return sum(z) / (2 * m)

    
    def fit(self, X, Y):
        m = len(Y)   
        lastJ = 0
        times = 0
        X = mat(X)  #转化为矩阵对象
        Y = mat(Y)
        X = hstack((ones((len(X), 1)), X)) #追加θ0对应的x值，统为1，变成m*(n+1)
        self.theta = zeros((shape(X)[1], 1))  # 定义为3*1的参数矩阵
  
        while  True:
            for i in range(m):
                d = (Y[i] - X[i]*self.theta)* X[i]
                self.theta += self.alpha * d.T
  
            #sum = (Y - X * self.theta).T * X
            #self.theta += self.alpha * sum.T / m
            J = self.computeCostJ(X, Y, self.theta)
            if abs(J - lastJ) < self.epsilon:  # 比较上一次与这次的costFun差值，来判断是否结束
                break
            lastJ = J
            times += 1
        self.times = times
        self.intercept_ = self.theta[0]
        self.coef_ = self.theta[1:]
        return self
    
    def predict(self,x):
        x = mat(x)
        x = hstack((ones((len(x), 1)), x)) 
        y = x * self.theta
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
