# -*- coding:utf-8 -*-
'''
Created on 2017年5月7日

@author: Jeffrey Zhou
'''

'''
soft-max多分类实现
'''

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression 
from userlib import MyLogisticRegression 
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def sigmoid(z):
    return(1.0 / (1.0 + np.exp(-z)))

def costFunction(theta, X, y):
#     theta = np.matrix(theta)
#     X = np.matrix(X)
#     y = np.matrix(y)
#     first = np.multiply(-y, np.log(sigmoid(X * theta)))
#     second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta)))
#     return np.sum(first - second) / (len(X))
    m = y.size
    h = sigmoid(X.dot(theta))
     
    J = -1.0*(1.0/m)*(np.log(h).T.dot(y)+np.log(1.0-h).T.dot(1.0-y))
                
    if np.isnan(J[0]):
        return(np.inf)
    return(J[0])

def gradient(theta, X, y):
    m = y.size
    h = sigmoid(X.dot(theta.reshape(-1,1)))
    
    grad =(1.0/m)*X.T.dot(h-y)

    return(grad.flatten())

if __name__ == '__main__':
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

    print (y ==1)
    
    y1 = y.copy()
    y1[y != 0] = 3
    mode1 = LogisticRegression()
    mode1.fit(x, y1)
    print mode1.coef_, mode1.intercept_
    
    y2 = y.copy()
    y2[y != 1] = 3
    mode2 = LogisticRegression()
    mode2.fit(x, y2)
    print mode2.coef_, mode2.intercept_
    
    y3 = y.copy()
    y3[y != 2] = 3
    mode3 = LogisticRegression()
    mode3.fit(x, y3)
    print mode3.coef_, mode3.intercept_
    
#     testx = [5.1,3.5]
#     print mode1.predict(testx)
#     print mode2.predict(testx)
#     print mode3.predict(testx)
    
    
    
    mode = MyLogisticRegression()
#     mode.fit(x, y1)
#     print mode.coef_,mode.times


    x =  np.hstack((np.ones((x.shape[0], 1)), x))
    initial_theta = np.zeros(x.shape[1])
    cost = costFunction(initial_theta,x, y1)
    grad = gradient(initial_theta, x, y1)
    print('Cost: \n', cost)
    print('Grad: \n', grad)
    
    res = minimize(costFunction, initial_theta, args=(x, y1), method=None, jac=gradient, options={'maxiter':100})
    print res
    print res.x

#     opti=Optimizer.Optimizer(1000,"SGD",1,2000,0.13,2000*0.99);    
#     opti.set_datasets(train,test,validate);
#     opti.set_functions(self.negative_log_likelihood,self.set_training_data,self.classify,self.callback,self.learn,None,None);
#     opti.run();


