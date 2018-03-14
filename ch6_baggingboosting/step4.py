# coding=utf-8
'''
Created on 2017年5月19日

@author: zjf
'''

'''
xgBoost
'''
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split   


if __name__ == '__main__':
    path = '..\\ch2_classification\\iris.data'  # 数据文件路径
    data = pd.read_csv(path, header=None)
    x = data[range(4)]
    y = pd.Categorical(data[4]).codes
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=50)

    data_train = xgb.DMatrix(x_train, label=y_train)
    data_test = xgb.DMatrix(x_test, label=y_test)
    watch_list = [(data_test, 'eval'), (data_train, 'train')]
    param = {'max_depth': 2, 'eta': 0.3, 'silent': 1, 'objective': 'multi:softmax', 'num_class': 3}

    mode = xgb.train(param, data_train, num_boost_round=6, evals=watch_list)
    y_hat = mode.predict(data_test)
    
    c = np.count_nonzero(y_test != y_hat)
    print '识别准确率：', (100 - (c * 100 / float(len(y)))), '%'
     