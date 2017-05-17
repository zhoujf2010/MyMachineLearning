# -*- coding:utf-8 -*-
'''
Created on 2017年5月17日

@author: Jeffrey Zhou
'''

'''
bagging
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

def f(x):
    return 0.5*np.exp(-(x+3) **2) + np.exp(-x**2) + 0.5*np.exp(-(x-3) ** 2)


if __name__ == '__main__':
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    
    #模拟数据，结合一函数，制造一些干扰
    np.random.seed(0)
    N = 200
    x = np.random.rand(N) * 10 - 5  # [-5,5)
    x = np.sort(x)
    y = f(x) + 0.05*np.random.randn(N)
    x.shape = -1, 1
    x_test = np.linspace(1.1*x.min(), 1.1*x.max(), 1000)
    
    #采用决策树效果
    mode = DecisionTreeRegressor(max_depth=5)
    
#     mode = RidgeCV(alphas=np.logspace(-3, 2, 20), fit_intercept=False)
#     mode = Pipeline([('poly', PolynomialFeatures(degree=6)), ('Ridge', mode)])
        
    #采用基于上面mode的bagging效果（100次)
    mode = BaggingRegressor(mode, n_estimators=100, max_samples=0.2)
    
    
    mode.fit(x,y)
    y_yat = mode.predict(x_test.reshape(-1, 1))
    
    plt.plot(x,y,'rx',label=u'训练数据')
    plt.plot(x_test,f(x_test),'g-',label=u'真实数据')
    plt.plot(x_test,y_yat.ravel(),'y-',label=u'拟合数据')
    plt.legend(loc='upper left')
    plt.show()