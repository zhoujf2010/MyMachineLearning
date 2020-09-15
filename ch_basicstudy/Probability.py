'''
Created on 2020年9月12日

@author: zjf
'''

import random
import numpy as np
from matplotlib import pyplot as plt
import operator as op
from functools import reduce


def bernoulli(p, k):  # 伯努利分布
    return p if k else 1 - p


def binomial(n, p, k):  # 二项分布
    q = 1 - p
    
    # calc  Cn pick k
    r = min(k, n - k)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    c = numer / denom
    
    y = c * (p ** k) * (q ** (n - k))
    return y


def categorical(p, k):  # 多伯努利分布
    return p[k]


def uniform(x, a, b):  # 均匀分布
    if a <= x and x <= b:
        y = 1 / (b - a)
    else:
        y = 0
    return y


def exponential(x, lamb):  # 指数分布
    y = lamb * np.exp(-lamb * x)
    return y


def gaussian(x, mu, sigma):  # 高斯分布
    a = ((x - mu) ** 2) / (2 * (sigma ** 2))
    y = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-a)
    return y


def normal(x):  # 正态分布
    return gaussian(x, 0, 1)


def gamma_function(n):  # gamma函数
    cal = 1
    for i in range(2, n):
        cal *= i
    return cal


def gamma(x, a, b):  # gamma分布
    c = (b ** a) / gamma_function(a)
    y = c * (x ** (a - 1)) * np.exp(-b * x)
    return y


def beta(x, a, b):  # beta 分布
    gamma = gamma_function(a + b) / \
            (gamma_function(a) * gamma_function(b))
    y = gamma * (x ** (a - 1)) * ((1 - x) ** (b - 1))
    return y

def chi_squared(x, k):#卡方分布
    c = 1 / (2 ** (k/2)) * gamma_function(k//2)
    y = c * (x ** (k/2 - 1)) * np.exp(-x /2)
    return y

def student_t(x, n): #t分布
    c = gamma_function((n + 1) // 2) \
        / np.sqrt(n * np.pi) * gamma_function(n // 2)
    y = c * (1 + x**2 / n) ** (-((n + 1) / 2))
    return y


x = np.arange(0, 40)
# # x = np.arange(-100,100)#均匀分布用
# x = (x - np.mean(x)) / np.std(x)  # 正则化
x = np.arange(-40, 40) #分布用

y = []
for k in x:
#     pick = bernoulli(0.7, k=bool(random.getrandbits(1)))
#     pick = binomial(len(x), 0.7, k)
#     pick= categorical([0.2,0.3,0.5],k=random.randint(0, 2))
#     pick = uniform(k,-50,30)
#     pick = exponential(k,0.5)
#     pick = gaussian(k, np.mean(x), np.std(x))
#     pick = gamma(k, 3, 1)
#     pick = beta(k,1,3)
#     pick = chi_squared(k,3)
    pick = student_t(k,2)
    y.append(pick)
# 
u, s = np.mean(y), np.std(y)
print("mean=", u, "std=", s)
# 
# # plt.scatter(x, y) #离散所用
plt.plot(x, y)  # 连续所用
plt.show()
