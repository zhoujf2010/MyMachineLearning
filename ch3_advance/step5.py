# -*- coding:utf-8 -*-
'''
Created on 2017年11月26日

连续数据处理
@author: Jeffrey Zhou
'''
#https://www.cnblogs.com/chaosimple/p/4153167.html


import warnings
from sklearn.exceptions import DataConversionWarning
from exceptions import DeprecationWarning
from sklearn import preprocessing
import numpy as np

if __name__ == '__main__':
    warnings.filterwarnings(action='ignore', category=DataConversionWarning)  # 去掉警告
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)  # 去掉警告
    np.set_printoptions(linewidth=400) #设置打印不换行
    #标准化
    x_old = np.array([1,2,3,4,5,6,7,8,9])
    print '原值',x_old
    x = preprocessing.scale(x_old)
    print '标准化后：',x
    print '平均值：',x.mean(),'标准差：',x.std()
     
    #库 使用该类的好处在于可以保存训练集中的参数（均值、方差）直接使用其对象转换测试集数据。
    scaler = preprocessing.StandardScaler().fit(x_old)
    x =scaler.transform(x_old)    
    print '用库标准化后：',x
    print '平均值：',x.mean(),'标准差：',x.std()
    print scaler.mean_,scaler.std_     

    #归一化
    #MinMaxScaler
    min_max_scaler = preprocessing.MinMaxScaler()
    x = min_max_scaler.fit_transform(x_old)
    print x
    print '求和：',x.sum(),'缩放因子：',min_max_scaler.scale_ 
    
    #正则化
    normalizer = preprocessing.Normalizer(norm='l2')#.fit(x_old)  # fit does nothing
    print normalizer
    x = normalizer.transform(x_old)   
    print x  
    
    

    
