# -*- coding:utf-8 -*-
'''
Created on 2017年12月12日

离散数据处理
@author: zjf
'''

from sklearn import preprocessing
import pandas as pd
import numpy as np


if __name__ == '__main__':
    # one-hot
    enc = preprocessing.OneHotEncoder()
    enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
    print enc.transform([[0, 1, 3]]).toarray()
    
    
    pd.set_option('display.width', 300)
    pd.set_option('display.max_columns', 300)

    data = pd.read_csv('car.data', header=None)
    data[0] = pd.Categorical(data[0]).codes
    n_columns = len(data.columns)
    columns = ['buy', 'maintain', 'doors', 'persons', 'boot', 'safety', 'accept']
    new_columns = dict(zip(np.arange(n_columns), columns))
    data.rename(columns=new_columns, inplace=True)
    print data.head(10)
    
    
    # one-hot编码
    x = pd.DataFrame()
    for col in columns[:-1]:
        t = pd.get_dummies(data[col])
        t = t.rename(columns=lambda x: col+'_'+str(x))
        x = pd.concat((x, t), axis=1)
    print x.head(10)
    # print x.columns
    y = pd.Categorical(data['accept']).codes
    print 'y=',y