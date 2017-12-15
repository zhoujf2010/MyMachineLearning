# coding=utf-8
'''
Created on 2017年12月13日

@author: zjf
正则项+交叉验证Cross-validation
'''

from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures

import warnings
from sklearn.exceptions import DataConversionWarning
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np;


if __name__ == '__main__':
    warnings.filterwarnings(action='ignore', category=DataConversionWarning)  # 去掉警告
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)  # 去掉警告
    np.set_printoptions(suppress=True,linewidth=300) #设置展示时不要用科学计数法
    np.random.seed(0)  # 设置随机种子，目的是为了每次都相同的随机数
    x = np.linspace(0, 6, 9)
    y = x ** 2 - 4 * x - 3 + np.random.randn(9)
    x.shape = -1, 1
    y.shape = -1, 1
    
    print 'x=', x.T, '\r\ny=', y.T
    
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.plot(x, y, 'rx', label=u'原始数据')
    
    #提升为多项式
    pmode = PolynomialFeatures(degree=8)
    x2 = pmode.fit_transform(x)
    #print x2
    
    #线性回归
    mode1 = LinearRegression()
    mode1.fit(x2,y)
    print mode1.coef_,mode1.intercept_
    x_hat = np.linspace(min(x), max(x), 100)
    x_hat.shape = -1, 1
    x_hat2 =  pmode.fit_transform(x_hat)
    y_hat = mode1.predict(x_hat2)
    plt.plot(x_hat, y_hat, label=u'线性回归')
    
    #Ridge回归
    #原：mode1 = Ridge(alpha=0.5)
    mode1 = Ridge()
    alpha = np.logspace(-3, 2, 100)
    mode1 = GridSearchCV(mode1, param_grid={'alpha':alpha}, cv=5)
    mode1.fit(x2,y)
    print '超参数:', mode1.best_params_
    x_hat = np.linspace(min(x), max(x), 100)
    x_hat.shape = -1, 1
    x_hat2 =  pmode.fit_transform(x_hat)
    y_hat = mode1.predict(x_hat2)
    plt.plot(x_hat, y_hat, label=u'Ridge回归')
    
    #Lasso
    mode1 = Lasso()
    alpha = np.logspace(-3, 2, 100)
    mode1 = GridSearchCV(mode1, param_grid={'alpha':alpha}, cv=5)
    mode1.fit(x2,y)
    print '超参数:', mode1.best_params_
    x_hat = np.linspace(min(x), max(x), 100)
    x_hat.shape = -1, 1
    x_hat2 =  pmode.fit_transform(x_hat)
    y_hat = mode1.predict(x_hat2)
    plt.plot(x_hat, y_hat, label=u'Lasso回归')
    
    #ElasticNet
    mode1 = ElasticNet()
    alpha = np.logspace(-3, 2, 100)
    ratio = [.1, .5, .7, .9, .95, .99, 1]
    mode1 = GridSearchCV(mode1, param_grid={'alpha':alpha, 'l1_ratio':ratio}, cv=5)
    mode1.fit(x2,y)
    print '超参数:', mode1.best_params_
    x_hat = np.linspace(min(x), max(x), 100)
    x_hat.shape = -1, 1
    x_hat2 =  pmode.fit_transform(x_hat)
    y_hat = mode1.predict(x_hat2)
    plt.plot(x_hat, y_hat, label=u'ElasticNet')
         
    plt.legend(loc="upper left")
    plt.show()
