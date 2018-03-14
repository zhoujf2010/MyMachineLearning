# coding=utf-8
'''
Created on 2017年5月8日

@author: zjf
'''

'''
调用库完成决策树
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.model_selection import train_test_split
# import pydotplus

if __name__ == '__main__':
    data = [[1, 1, 'yes'],
              [1, 1, 'yes'],
              [1, 0, 'no'],
              [0, 1, 'no'],
              [0, 1, 'no']]

    x, y = np.split(data, (2,), axis=1)
    
    model = DecisionTreeClassifier(criterion='entropy') #entropy or gini
    model.fit(x, y)
    
    print model.predict([1,1])
    print model.predict([0,1])
    
     
#     # 保存
#     # 1、输出
#     with open('iris.dot', 'w') as f:
#         tree.export_graphviz(model, out_file=f)
#     # 2、给定文件名
    tree.export_graphviz(model, out_file='test.dot')
#     # 3、输出为pdf格式
#     dot_data = tree.export_graphviz(model, out_file=None,
#                                     filled=True, rounded=True, special_characters=True)
#     graph = pydotplus.graph_from_dot_data(dot_data)
#     graph.write_pdf('iris.pdf')
#     f = open('iris.png', 'wb')
#     f.write(graph.create_png())
#     f.close()


    #拟合示例
    x = np.linspace(0, 10, 100)
    y = np.sin(x)+ np.random.randn(100) * 0.05
    x = x.reshape((-1,1))
    y = y.reshape((-1,1))
    
    #max_depth 为最大分支深度，防过拟合
    model = DecisionTreeRegressor(criterion='mse', max_depth=5)
    model.fit(x,y)
    
    y_hat = model.predict(x)
    
    plt.plot(x,y,'rx')
    plt.plot(x,y_hat,'g-')
    plt.show()
    
