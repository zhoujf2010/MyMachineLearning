# coding=utf-8
'''
Created on 2018年3月16日

@author: zjf
'''

import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn import preprocessing

if __name__ == '__main__':
    y=["Firefox","Chrome","Safari","IE"]
#     print y
#     enc = OneHotEncoder()  
#     enc.fit(y)  
      
    #array = enc.transform('Chrome').toarray()  


    enc = preprocessing.OneHotEncoder()
    enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])    # fit来学习编码
    print enc.transform([[0, 1, 3]]).toarray()    # 进行编码
