# -*- coding:utf-8 -*-
'''
Created on 2017年5月7日

@author: Jeffrey Zhou
'''

'''
利用minimize库进行梯度下降
'''

import numpy as np
from scipy.optimize import minimize


class MyLogisticRegression2(object):
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
