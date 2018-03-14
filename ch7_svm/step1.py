# -*- coding:utf-8 -*-
'''
Created on 2017年5月30日

@author: Jeffrey Zhou
'''

'''
SVM对鸢尾花数据分类
'''

import pandas as pd
from sklearn import svm
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


if __name__ == '__main__':
    path = '..\\ch2_classification\\iris.data'  # 数据文件路径
    data = pd.read_csv(path, header=None)
    x, y = data[range(4)], data[4]
    y = pd.Categorical(y).codes
    x = x[[0, 1]] #取花萼长度，花萼宽度 两个属性
    
    #建模
    mode = svm.SVC(C=0.1,kernel='linear', decision_function_shape='ovr')
    mode.fit(x,y)
    
    # 画图
    x1_min, x2_min = x.min()
    x1_max, x2_max = x.max()
    x1, x2 = np.mgrid[x1_min:x1_max:500j, x2_min:x2_max:500j]  # 生成网格采样点
    grid_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点
    grid_hat = mode.predict(grid_test)       # 预测分类值
    grid_hat = grid_hat.reshape(x1.shape)  # 使之与输入的形状相同

    cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
    plt.figure(facecolor='w')
    plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)
    plt.scatter(x[0], x[1], c=y, edgecolors='k', s=50, cmap=cm_dark)      # 样本
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.grid(b=True, ls=':')
    plt.show()

    