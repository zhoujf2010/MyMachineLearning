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
from scipy.optimize import minimize


class MyLogisticRegression(object):
    def __init__(self, alpha=0.01, epsilon=0.00001):
        self.alpha = alpha
        self.epsilon = epsilon  
     
    def sigmod(self, inX):
        return 1.0 / (1 + np.exp(-inX))   
    
    def costFunction(self, theta, X, y):
        m = y.size
        h = self.sigmod(X.dot(theta))
        
        J = -1.0 * (1.0 / m) * (np.log(h).T.dot(y) + np.log(1.0 - h).T.dot(1.0 - y))

        if np.isnan(J[0]):
            return(np.inf)
        return(J[0])
    
    def gradient(self, theta, X, y):
        m = y.size
        h = self.sigmod(X.dot(theta.reshape(-1, 1)))
        
        grad = (1.0 / m) * X.T.dot(h - y)
    
        return(grad.flatten())

    def fit(self, X, Y):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        
        initial_theta = np.zeros(X.shape[1])
#         cost = self.costFunction(initial_theta, X, Y)
#         grad = self.gradient(initial_theta, X, Y)
        res = minimize(self.costFunction, initial_theta, args=(X, Y),
                       method=None, jac=self.gradient, options={'maxiter':100})

        self.theta = res.x
        self.intercept_ = res.x[0]
        self.coef_ = self.theta[1:]
        return self

    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
#         x = mat(t)
        Y = self.sigmod(X.dot(self.theta))
#         h = self.sigmod(x * (self.theta.T))
        # Y = (-self.theta[0] - self.theta[1] * x) / self.theta[2]
        return Y
    
if __name__ == '__main__':
#     data = pd.read_csv('testdata1.txt', header=None)
    data = pd.read_csv('testdata1.txt', header=None)
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
    
    
    
