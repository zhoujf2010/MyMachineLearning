# coding=utf-8
'''
Created on 2017年12月28日

@author: zjf
'''

'''
决策树做拟合
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

if __name__ == '__main__':
    N = 100
    x = np.random.rand(N) * 6 - 3  # [-3,3)
    x.sort()
    y = np.sin(x) + np.random.randn(N) * 0.05
    print y
    x = x.reshape(-1, 1)  # 转置后，得到N个样本，每个样本都是1维的
    print x
    
    plt.plot(x, y, 'ro', ms=2, label='Actual')
    for deep in [2, 4, 6, 9]:  # 显示不同深度的树拟合的效果
        dt = DecisionTreeRegressor(criterion='mse', max_depth=deep)
        dt.fit(x, y)
        x_test = np.linspace(-3, 3, 50).reshape(-1, 1)
        y_hat = dt.predict(x_test)
        plt.plot(x_test, y_hat, linewidth=1, label='Predict deep=%d' % deep)
    plt.legend(loc='upper left')
    plt.grid(True, ls=":")
    plt.show()




