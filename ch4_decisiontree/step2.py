# coding=utf-8
'''
Created on 2017年5月8日

@author: zjf
'''

'''
实现决策树
'''

import numpy as np
from math import log

# 计算y的信息熵
def calcEntripy(y):
    n = len(y)
    labelcount = {} 
    # 将列表装入hash表中，key名值，value为数量
    for item in y:
        item = item[0]
        if item not in labelcount.keys():
            labelcount[item] = 0
        labelcount[item] += 1
    # 计算每一项plog(p)，再求和
    ret = 0
    for key in labelcount:
        prob = float(labelcount[key]) / n
        ret -= prob * np.log2(prob)
    return ret

#拆分数据集
def splitdataset(x, y, axis, value):
    retdataset = []
    retyset = []
    for i, row in enumerate(x):
        if (row[axis] == value):
            if axis == 0:
                reducerow = row[axis + 1:]
            elif axis == len(x[0]) - 1:
                reducerow = row[:axis]
            else:
                reducerow = np.stack((row[:axis], row[axis + 1:]), axis=1)
            retdataset.append(reducerow.tolist())
            retyset.append(y[i].tolist())
    return retdataset, retyset

def choosebestFeatureToSplit(x, y):
    n = len(x[0])  # 特征数量
    bestentropy = calcEntripy(y)
    bestinfoGain = 0.0;
    bestFeature = -1;
    for i in range(n):
        feetList = [example[i] for example in x]
        uniqueval = set(feetList)
        newentropy = 0.0
        for value in uniqueval:
            subxset, subyset = splitdataset(x, y, i, value)
            prob = len(subxset) / float(len(x))
            newentropy += prob * calcEntripy(subyset)
        infogan = bestentropy - newentropy
        if infogan > bestinfoGain:
            bestinfoGain = infogan
            bestFeature = i;
    return bestFeature
    

def splitdataset2(dataset, axis, value):
    retdataset = []
    for row in dataset:
        if (row[axis] == value):
            reducerow = row[:axis]
            reducerow.extend(row[axis + 1:])
            retdataset.append(reducerow)
    return retdataset

if __name__ == '__main__':
    data = [[1, 1, 'yes'],
              [1, 1, 'yes'],
              [1, 0, 'no'],
              [0, 1, 'no'],
              [0, 1, 'no']]

    
    x, y = np.split(data, (2,), axis=1)
    
    print splitdataset(x, y, 0, '1')
    
    print calcEntripy(y)
#     
    print choosebestFeatureToSplit(x, y)
#     
