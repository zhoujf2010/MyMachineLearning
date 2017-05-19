# coding=utf-8
'''
Created on 2017年5月19日

@author: zjf
'''

'''
GBDT实验,鸢尾花数据
'''

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier 

if __name__ == '__main__':
    path = '..\\ch2_classification\\iris.data'  # 数据文件路径
    data = pd.read_csv(path, header=None)
    x = data[range(4)]
    y = pd.Categorical(data[4]).codes
    
    mode = GradientBoostingClassifier(n_estimators=20, learning_rate=1.0,max_depth=1, random_state=0)  # 迭代20次  
    mode.fit(x, y)
    
    y_hat = mode.predict(x)
    
    c = np.count_nonzero(y != y_hat)
    print '识别准确率：', (100 - (c * 100 / float(len(y)))), '%'
    
