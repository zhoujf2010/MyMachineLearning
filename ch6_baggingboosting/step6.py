# -*- coding:utf-8 -*-
'''
Created on 2017年5月26日

@author: Jeffrey Zhou
'''


'''
Adboost实现
'''

import numpy as np

def stumClassify(x, threshVal, threshIneq):
    retArr = np.ones((np.shape(x)[0], 1))
    if threshIneq == 'lt':
        retArr[x <= threshVal] = -1.0
    else:
        retArr[x > threshVal] = -1.0
    return retArr

# 建立树桩(Stump)
def buildStump(x, y, D):
    m = np.shape(x)[0]
    bestStump = {}
    bestClasEst = np.mat(np.zeros((m, 1)))
    minEror = np.Inf
    for threshVal in np.arange(0.5, 9.5, 1):
        for inequal in ['lt', 'gt']:
            predictedVals = stumClassify(x, threshVal, inequal)
            errArr = np.mat(np.ones((m, 1)))
            errArr[predictedVals == y] = 0
            weightError = D.T * errArr
            if weightError < minEror:
                minEror = weightError
                bestClasEst = predictedVals.copy()
                bestStump['thresh'] = threshVal
                bestStump['ineq'] = inequal
    return bestStump, minEror, bestClasEst.T
                    
def adaBoostfit(x, y, numIt=40):
    weekClassArr = []
    m = np.shape(x)[0]
    D = np.mat(np.ones((m, 1)) / m)
    aggClassEst = np.mat(np.zeros((m, 1)))
    for k in range(numIt):
        bestStump, error, classEst = buildStump(x, y, D)
        print "D%d:" % k, D.T
        gamma = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
        bestStump['gamma'] = gamma
        print "alpha:", bestStump["ineq"], bestStump["thresh"] , '   gamma值：', gamma
        weekClassArr.append(bestStump)
        print '分类y：', classEst
        expon = np.multiply(-1 * gamma * y, classEst.T)
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()
        aggClassEst += gamma * classEst.T
        
        aggErrors = np.multiply(np.sign(aggClassEst) != y, np.ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print "错误率：", errorRate, "\n"
        if errorRate == 0.0:break
    return weekClassArr 

def addboostpredict(x, classArr):
    m = np.shape(x)[0]
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(len(classArr)):
        classEst = stumClassify(x, classArr[i]["thresh"], classArr[i]["ineq"])
        aggClassEst += classArr[i]["gamma"] * classEst
    return np.sign(aggClassEst)
    
if __name__ == '__main__':
    np.set_printoptions(linewidth=300)
    
    data = np.matrix([[0.0, 1.0], [1, 1.0], [2, 1.0], [3, -1.0], [4, -1.0], [5, -1.0], [6, 1.0], [7, 1.0], [8, 1.0], [9, -1.0]])
    x, y = np.split(data, (1,), axis=1)
    
    # test tree stump
    D = np.mat(np.ones((len(x), 1)) / len(x))
    print "test Tree Stump:", buildStump(x, y, D), "\n"
    
    classArr = adaBoostfit(x, y, 9)

    y_hat = addboostpredict(x, classArr)
    print "y_hat=", y_hat.T     
            
