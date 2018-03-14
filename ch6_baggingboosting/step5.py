# coding=utf-8
'''
Created on 2017年3月27日

@author: zjf
'''

'''
本例尝试一个概率为0.6的事件，在重复100次后的概率
bagging
'''

import numpy as np
import operator
import matplotlib.pyplot as plt

def c(n, k):
    return reduce(operator.mul, range(n - k + 1, n + 1)) / reduce(operator.mul, range(1, k + 1))


def bagging(n, p):
    s = 0;
    for i in range(n / 2 + 1, n + 1):
        s += c(n, i) * p ** i * (1 - p) ** (n - i)
    return s

if __name__ == '__main__':
    n = 100
    x = np.arange(1, n, 2)
    y = np.zeros_like(x, dtype=np.float)
    for i, t in enumerate(x):
        y[i] = bagging(t, 0.6)
        if i % 10 == 0:
            print t, '采样正确率：', y[i]

    # 显示结果
    plt.plot(x, y, 'ro-')
    plt.grid(True)
    plt.show()
