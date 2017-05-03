# -*- coding:utf-8 -*-
'''
Created on 2017年5月3日

@author: Jeffrey Zhou
'''

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches

if __name__ == '__main__':
    data = pd.read_csv('testdata1.txt', header=None)
    x, y = np.split(data.values, (2,), axis=1)
#     print data.head(10)
#     print 'x=\r\n',x,'\r\ny=\r\n',y

#     print 
#     print x[:,1]

    mode = LogisticRegression()
    mode.fit(x, y)
   
    # 用背景图色展示预测数据
    N, M = 500, 500  # x,y上切分多细
    t0 = np.linspace(x[:, 0].min(), x[:, 0].max(), N)
    t1 = np.linspace(x[:, 1].min(), x[:, 1].max(), N)
    x0, x1 = np.meshgrid(t0, t1)  # 填统到二维数组中
    
    x_test = np.stack((x0.flat, x1.flat), axis=1)
    y_hat = mode.predict(x_test).reshape(x0.shape)
    
    plt.pcolormesh(x0,x1,y_hat,cmap=mpl.colors.ListedColormap(['#77E0A0', '#FF8080']))
    
    
    # 展示原始数据
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=mpl.colors.ListedColormap(['g', 'b']))
    
    plt.show()
    
    
    
