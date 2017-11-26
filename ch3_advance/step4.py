# -*- coding:utf-8 -*-
'''
Created on 2017年5月2日

@author: Jeffrey Zhou
'''

'''
局部加权拟合
'''

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


if __name__ == '__main__':
    # 读取数据
    data = pd.read_csv('ex0.txt', sep='\t', header=None)
    x, y = np.split(data.values, (2,), axis=1)
    X = np.mat(x); Y = np.mat(y)  # 转为矩阵
    
    # 传统正规方程拟合
    theta = (X.T * X).I * X.T * Y
    y_hat1 = X * theta
    
    # 正规则方程中加入局部权重
    m = np.shape(X)[0]
    y_hat2 = np.zeros(m)
    k = 0.003
    weights = np.mat(np.eye((m)))  
    for i in range(m):
        for j in range(m): 
            diffMat = X[i, :] - X[j, :]
            weights[j, j] = np.exp(diffMat * diffMat.T / (-2.0 * k ** 2))
        theta = (X.T * (weights * X)).I * (X.T * (weights * Y))
        y_hat2[i] = X[i, :] * theta
    
    # 根据X排序
    strInd = X[:, 1].argsort(0).flatten().A[0]
    xSort = X[strInd]
    y_hat2 = y_hat2[strInd]
    
    # 图形显示
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.plot(x[:, 1], y, 'rx', label=u'原始数据')
    plt.plot(x[:, 1], y_hat1, 'b-', label=u'普通拟合')
    plt.plot(xSort[:, 1], y_hat2, 'g-', label=u'带权拟合')
    plt.legend(loc="upper left")
    plt.show()

