# -*- coding:utf-8 -*-
'''
Created on 2017年4月27日

@author: Jeffrey Zhou
'''

import pandas as pd;
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np;
from sklearn.linear_model import LinearRegression


def not_empty(s):
    return s != ''

if __name__ == '__main__':
    np.set_printoptions(suppress=True)  # 控制print时，不要用科学计数法显示
    datard = pd.read_csv("housing.data", header=None)
    data = np.empty((len(datard), 14))  # 创建一个N行14列的数据
    for i, row in enumerate(datard.values):
        data[i] = map(float, filter(not_empty, row[0].split(' ')))  # 处理一行数据，拆分到数组中
    
#     print "data=\n",data
    x = data[:, 7:8]
    y = data[:, 13:14]
    
#     print 'x=', x
#     print 'y=', y

#     # 线性回归
    mode = LinearRegression()
    mode.fit(x, y)
    print mode.coef_, mode.intercept_
    
    y_hat = mode.predict(x)  # 用模型直接预测数据
    
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.plot(x, y, 'rx', label=u'原始数据')
    plt.plot(x, y_hat, 'g-', label=u'预测数据')
    plt.legend(loc="upper right")
    plt.show()
