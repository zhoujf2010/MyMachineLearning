# -*- coding:utf-8 -*-
'''
Created on 2017年5月17日

@author: Jeffrey Zhou
'''

'''
随机森林处理鸢尾花数据
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':
    # 花萼长度、花萼宽度，花瓣长度，花瓣宽度
    iris_feature_E = 'sepal length', 'sepal width', 'petal length', 'petal width'
    iris_feature = u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度'
    iris_class = 'Iris-setosa', 'Iris-versicolor', 'Iris-virginica'
    

    path = '..\\ch2_classification\\iris.data'  # 数据文件路径
    data = pd.read_csv(path, header=None)
    x = data[range(4)]
    y = pd.Categorical(data[4]).codes
    # 为了可视化，仅使用前两列特征
    x = x.iloc[:, :2]
    
    
    model = RandomForestClassifier(n_estimators=200, criterion='entropy', max_depth=3)
    model.fit(x, y)
    y_hat = model.predict(x)  # 测试数据
    
    c = np.count_nonzero(y_hat == y)  # 计算预测对的数量
    print '精度为：', (c * 100.0 / len(y)),'%'
    
