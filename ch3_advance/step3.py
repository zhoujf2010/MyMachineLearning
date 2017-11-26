# -*- coding:utf-8 -*-
'''
Created on 2017年5月2日

@author: Jeffrey Zhou
'''

'''
多项式过拟合示例
'''

from sklearn.preprocessing import PolynomialFeatures
import pandas as pd;
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np;
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


if __name__ == '__main__':
    np.random.seed(0)  # 设置随机种子，目的是为了每次都相同的随机数
    x = np.linspace(0, 6, 9)
    y = x ** 2 - 4 * x - 3 + np.random.randn(9)
    x.shape = -1, 1
    y.shape = -1, 1
    
    print 'x=', x.T, '\r\ny=', y.T
        
    mode = Pipeline([
        ('poly', PolynomialFeatures()),
        ('linear', LinearRegression(fit_intercept=False))
        ])
    
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.plot(x, y, 'rx', label=u'原始数据')
    
    for d in range(1, 8):
        mode.set_params(poly__degree=d)
        mode.fit(x, y)
        
        x_hat = np.linspace(min(x), max(x), 100)
        x_hat.shape = -1, 1
        y_hat = mode.predict(x_hat)
        plt.plot(x_hat, y_hat, label=u'd=%d阶' % d)
        
    plt.legend(loc="upper left")
    plt.show()
