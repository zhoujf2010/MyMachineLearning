# -*- coding:utf-8 -*-
'''
Created on 2017年11月26日
梯度下降 辅助理解
@author: Jeffrey Zhou
'''

from numpy import *
import pandas as pd;
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np;
from mpl_toolkits.mplot3d import Axes3D  
from matplotlib import cm  
from step4 import gradientDescent

if __name__ == '__main__':
    data = pd.read_csv("ex1data1.txt", header=None)
    x, y = np.split(data.values, (1,), axis=1)
    X = np.mat(x); Y = np.mat(y)  # 转为矩阵
    
    mode = gradientDescent(alpha=0.01, epsilon=0.000001)
    mode.fit(x, y)
    print mode.coef_, mode.intercept_
    X = hstack((ones((len(X), 1)), X))  # 追加θ0对应的x值，统为1，变成m*(n+1)
    
    theta0_vals = np.arange(-10, 10, 0.2)  # 定义θ0值域[-10,10]
    theta1_vals = np.arange(-1, 4, 0.05)  # 定义θ1值域[-1,4]
    J_vals = zeros((len(theta0_vals), len(theta1_vals)))  # 定义出J(θ)的二维数组
     
    for i in range(len(theta0_vals)):
        for j in range(len(theta1_vals)):
            t = [theta0_vals[i], theta1_vals[j]]
            J_vals[j, i] = mode.computeCostJ(X, Y, mat(t).T);
    theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)  # 生成网格采样点
     
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(theta0_vals, theta1_vals, J_vals, rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=False)
    ax.view_init(elev=20., azim=225)
    ax.set_xlabel('theta0')
    ax.set_ylabel('theta1')
    ax.set_zlabel('J(theta)')
     
    ax = fig.add_subplot(122)
    ax.contour(theta0_vals, theta1_vals, J_vals, 50)  # 显示J(θ)等高线
    ax.plot(mode.intercept_, mode.coef_[0], 'xr') #将最低值显示出，并用红色叉显示
    ax.set_xlabel('theta0')
    ax.set_ylabel('theta1')
    
    plt.show()
    
