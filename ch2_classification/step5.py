# -*- coding:utf-8 -*-
'''
Created on 2017年5月7日

@author: Jeffrey Zhou
'''

'''
soft-max多分类实现
'''

from numpy import *
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression 
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import minimize


class MyLogisticRegression(object):
    def __init__(self, K=2, alpha=0.01, epsilon=0.00001):
        self.alpha = alpha
        self.epsilon = epsilon  
        self.K = K;
     
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
        self.theta = zeros((self.K, shape(X)[1]))  # 定义为K*3的参数矩阵
        tmptheta = zeros((self.K, shape(X)[1]))
        for times in range(100):
            for i in range(m):
                for k in range(self.K):
                    y = 0
                    if Y[i] == k:
                        y = 1 
                    p = self.predict2(X[i], k)
                    tmptheta[k] = self.theta[k] + self.alpha *((y - p) * X[i])# + 0.01 * self.theta[k])
                self.theta = tmptheta.copy()
            
#             sum = (Y - self.sigmod(X * self.theta)).T * X  # 与回归多了一个sigmod函数
#             self.theta += self.alpha * sum.T / m
#             J = self.computeCostJ(X, Y, self.theta)
#             if abs(J - lastJ) < self.epsilon:  # 比较上一次与这次的costFun差值，来判断是否结束
#                 break
#             lastJ = J
#             times += 1
        self.times = times
        self.intercept_ = self.theta
        self.coef_ = self.theta
        return self

    def predict2(self, X, k):
        c = np.exp(self.theta[k, :] * X.T)
        call = np.sum(np.exp(self.theta * X.T));
        p = c / call
        return p

    def predict(self, X):
        t = np.hstack((np.ones((X.shape[0], 1)), X))
        X = mat(t)
        m = len(X)   
        Y = zeros((m))
        for i in range(m):
            p = zeros(self.K)
            maxk = -1;
            maxP = -1;
            for k in range(self.K):
                p[k] = self.predict2(X[i], k) 
                if p[k] > maxP:
                    maxk = k                
                    maxP = p[k]
            Y[i] = maxk    
           
        return Y

if __name__ == '__main__':
    
    print np.log(np.exp(80)+np.exp(90))
    
    
    data = pd.read_csv("iris.data", header=None)
    data[4] = pd.Categorical(data[4]).codes
    x, y = np.split(data.values, (4,), axis=1)
    # 为了方便展示仅使用前两列特征
    x = x[:, :2]
#     print data.head(5)
#     print y
    
#     for i,t in enumerate(y):
#         if t !=2:
#             y[i]=3

#     print (y ==1)
#     
#     y1 = y.copy()
#     y1[y != 0] = 3
#     mode1 = LogisticRegression()
#     mode1.fit(x, y1)
#     print mode1.coef_, mode1.intercept_
#     
#     y2 = y.copy()
#     y2[y != 1] = 3
#     mode2 = LogisticRegression()
#     mode2.fit(x, y2)
#     print mode2.coef_, mode2.intercept_
#     
#     y3 = y.copy()
#     y3[y != 2] = 3
#     mode3 = LogisticRegression()
#     mode3.fit(x, y3)
#     print mode3.coef_, mode3.intercept_
    
#     testx = [5.1,3.5]
#     print mode1.predict(testx)
#     print mode2.predict(testx)
#     print mode3.predict(testx)
    
    
    
    mode = MyLogisticRegression(K=3)
    mode.fit(x, y)
    
    # x = mat(x)
    # mode.predict(x[0])
    # mode.predict(x[4])
    # mode.predict(x[50])
    
#     print mode.coef_,mode.times


#     x =  np.hstack((np.ones((x.shape[0], 1)), x))
#     initial_theta = np.zeros(x.shape[1])
#     cost = costFunction(initial_theta,x, y1)
#     grad = gradient(initial_theta, x, y1)
#     print('Cost: \n', cost)
#     print('Grad: \n', grad)
#     
#     res = minimize(costFunction, initial_theta, args=(x, y1), method=None, jac=gradient, options={'maxiter':100})
#     print res
#     print res.x

#     opti=Optimizer.Optimizer(1000,"SGD",1,2000,0.13,2000*0.99);    
#     opti.set_datasets(train,test,validate);
#     opti.set_functions(self.negative_log_likelihood,self.set_training_data,self.classify,self.callback,self.learn,None,None);
#     opti.run();

   
    # 用背景图色展示预测数据
    N, M = 50, 50  # x,y上切分多细
    t0 = np.linspace(x[:, 0].min(), x[:, 0].max(), N)
    t1 = np.linspace(x[:, 1].min(), x[:, 1].max(), N)
    x0, x1 = np.meshgrid(t0, t1)  # 填统到二维数组中
    
    x_test = np.stack((x0.flat, x1.flat), axis=1)
    y_hat = mode.predict(x_test).reshape(x0.shape)
    
    plt.pcolormesh(x0, x1, y_hat, cmap=mpl.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF']))
    
    # 展示原始数据
    plt.scatter(x[:, 0], x[:, 1], c=y[:, 0], cmap=mpl.colors.ListedColormap(['g', 'r', 'b']))
    
    plt.show()
    
