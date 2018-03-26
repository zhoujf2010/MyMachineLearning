# -*- coding:utf-8 -*-
'''
Created on 2017年5月2日

@author: Jeffrey Zhou
'''

'''
多项式
'''

from sklearn.preprocessing import PolynomialFeatures
import pandas as pd;
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np;
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


if __name__ == '__main__':
    np.set_printoptions(suppress=True) #设置展示时不要用科学计数法
    
    x = np.array([1, 2, 3, 4, 5, 6]).reshape((-1, 1))
    y = np.array([1, 4, 9, 16, 25, 36]).reshape((-1, 1))
    x_test = np.linspace(min(x), max(x), 100).reshape((-1, 1))

    #升成多项式(手动计算)
#     x2 = np.hstack((np.ones((x.shape[0], 1)), x, x * x))
#     x_test2 = np.hstack((np.ones((x_test.shape[0], 1)), x_test, x_test * x_test))
#     print (x2)

    #升成多项式(公式计算)
#     mode = PolynomialFeatures()
#     mode.set_params(degree=2)
#     x2 = mode.fit_transform(x)
#     x_test2 =mode.fit_transform(x_test)
#     print x2
    
    # 线性回归
#     mode = LinearRegression()
#     mode.fit(x2, y)
#     print 'theta=', mode.coef_, mode.intercept_
#     y_hat = mode.predict(x_test2)  # 用模型直接预测数据

    #用Pipeline合并两个模型
    mode = Pipeline([
        ('poly', PolynomialFeatures()),
        ('linear', LinearRegression())])
    mode.set_params(poly__degree=2)
    mode.fit(x, y)
    y_hat = mode.predict(x_test)  # 用模型直接预测数据

    #展示结果
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.plot(x, y, 'rx', label=u'原始数据')
    plt.plot(x_test, y_hat, 'g-', label=u'预测数据')
    plt.legend(loc="upper left")
    plt.show()
