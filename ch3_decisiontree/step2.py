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

def calcShannonEnt(dataset):
    n = len(dataset)
    labelcount = {}
    for row in dataset:
        currentLable = row[-1]
        if currentLable not in labelcount.keys():
            labelcount[currentLable] = 0
        labelcount[currentLable] += 1
    ret = 0
    for key in labelcount:
        prob = float(labelcount[key]) / n
        ret -= prob * np.log2(prob)
    return ret

def splitdataset(dataset,axis,value):
    retdataset=[]
    for row in dataset:
        if (row[axis] == value):
            reducerow = row[:axis]
            reducerow.extend(row[axis+1:])
            retdataset.append(reducerow)
    return retdataset

def choosebestFeatureToSplit(dataset):
    n = len(dataset[0])-1   #特征数量
    bestentropy = calcShannonEnt(dataset)
    bestinfoGain =0.0;
    bestFeature=-1;
    for i in range(n):
        feetList = [example[i] for example in dataset]
        uniqueval = set(feetList)
        newentropy =0.0
        for value in uniqueval:
            subdataset = splitdataset(dataset, i, value)
            prob = len(subdataset)/float(len(dataset))
            newentropy += prob * calcShannonEnt(subdataset)
        infogan = bestentropy- newentropy
        if infogan > bestinfoGain:
            bestinfoGain = infogan
            bestFeature=i;
    return bestFeature
    

if __name__ == '__main__':
    testdt = [[1,1,'yes'],
              [1,1,'yes'],
              [1,0,'no'],
              [0,1,'no'],
              [0,1,'no']]
    print calcShannonEnt(testdt)
    
    print choosebestFeatureToSplit(testdt)
    
