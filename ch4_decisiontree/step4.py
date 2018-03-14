# coding=utf-8
'''
Created on 2017年12月26日

@author: zjf
'''

'''
比较熵与Gini的误差
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

if __name__ == '__main__':
    x = np.arange(0.0001, 1, 0.0001, dtype=np.float)
    gini = 2 * x * (1 - x)
    entroy = -(x * np.log2(x)+(1-x)*np.log2(1-x))/2
    err = 1- np.max(np.vstack((x,1-x)),0)
    plt.plot(x, entroy, 'b-',label='entroy')
    plt.plot(x, gini, 'r-',label='gini')
    plt.plot(x, err, 'g--',label='Error')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()
