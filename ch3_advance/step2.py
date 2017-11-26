# -*- coding:utf-8 -*-
'''
Created on 2017年5月2日

@author: Jeffrey Zhou
'''

'''
分类多项式
'''

from sklearn.preprocessing import PolynomialFeatures
import pandas as pd;
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np;
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


if __name__ == '__main__':
    
    data = pd.read_csv("../ch2_classification/iris.data", header=None)
    data[4] = pd.Categorical(data[4]).codes
    x, y = np.split(data.values, (4,), axis=1)
    # 为了方便展示仅使用前两列特征
    x = x[:, :2]
#     print data.head(5)
#     print y

    #增加多项式维度
    pmode = PolynomialFeatures()
    pmode.set_params(degree=2)
    x2 = pmode.fit_transform(x)

    mode = LogisticRegression()
    mode.fit(x2, y)
    print mode.coef_, mode.intercept_
    
   
    # 用背景图色展示预测数据
    N, M = 500, 500  # x,y上切分多细
    t0 = np.linspace(x[:, 0].min(), x[:, 0].max(), N)
    t1 = np.linspace(x[:, 1].min(), x[:, 1].max(), N)
    x0, x1 = np.meshgrid(t0, t1)  # 填统到二维数组中
    
    x_test = np.stack((x0.flat, x1.flat), axis=1)
    x_test2 = pmode.fit_transform(x_test) #升维
    y_hat = mode.predict(x_test2).reshape(x0.shape)
    
    plt.pcolormesh(x0, x1, y_hat, cmap=mpl.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF']))
    
    # 展示原始数据
    plt.scatter(x[:, 0], x[:, 1], c=y[:, 0], cmap=mpl.colors.ListedColormap(['g', 'r', 'b']))
    
    plt.show()