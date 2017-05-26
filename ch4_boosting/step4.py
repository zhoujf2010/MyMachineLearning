# -*- coding:utf-8 -*-
'''
Created on 2017年5月26日

@author: Jeffrey Zhou
'''


'''
Adboost实现
'''

import numpy as np
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor

def stumClassify(x, dime, threshVal, threshIneq):
    retArr = np.ones((np.shape(x)[0], 1))
    if threshIneq == 'lt':
        retArr[x[:, dime] <= threshVal] = 1.0
    else:
        retArr[x[:, dime] > threshVal] = -1.0
    return retArr

def buildStump(x, y, D):
    m, n = np.shape(x)
    numStep = 10.0
    bestStump = {}
    bestClasEst = np.mat(np.zeros((m, 1)))
    minEror = np.Inf
    for i in range(n):
        rangeMin = x[:, i].min()
        rangeMax = x[:, i].max()
        stepSize = (rangeMax - rangeMin) / numStep
        for j in range(-1, int(numStep) + 1):
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumClassify(x, i, threshVal, inequal)
                errArr = np.mat(np.ones((m, 1)))
                errArr[predictedVals == y] = 0
                weightError = D.T * errArr
                # print 'split: dim %d,thresh %.2f,thresh inequal: %s,the wieght error is %.3f' % (i, threshVal, inequal, weightError)
                if weightError < minEror:
                    minEror = weightError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minEror, bestClasEst
                    
def adaBoostTrainDS(x, y, numIt=40):
    weekClassArr = []
    m = np.shape(x)[0]
    D = np.mat(np.ones((m, 1)) / m)
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(x, y, D)
        print "D:", D.T
        # error = np.array(error)[0][0]
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weekClassArr.append(bestStump)
        print 'classEst:', classEst.T
        expon = np.multiply(-1 * alpha * y, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()
        aggClassEst += alpha * classEst
        print "aggClassEst:", aggClassEst.T
        aggErrors = np.multiply(np.sign(aggClassEst) != y, np.ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print "total Error:", errorRate, "\n"
        if errorRate == 0.0:break
    return weekClassArr 

def addClassify(x, classArr):
    m = np.shape(x)[0]
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(len(classArr)):
        classEst = stumClassify(x, classArr[i]["dim"], classArr[i]["thresh"], classArr[i]["ineq"])
        aggClassEst += classArr[i]["alpha"] * classEst
        print aggClassEst
    return np.sign(aggClassEst)
        

if __name__ == '__main__':
    np.set_printoptions(linewidth=300)
    data = np.matrix([[0, 1], [1, 1], [2, 1], [3, -1], [4, -1], [5, -1], [6, 1], [7, 1], [8, 1], [9, -1]])
#     data = np.matrix([[1., 2.1, 1.0], [2., 1.1, 1.0], [1.3, 1., -1.0], [1., 1., -1.0], [2., 1., 1.0]])
    x, y = np.split(data, (1,), axis=1)
    
    model = DecisionTreeClassifier(criterion='entropy',max_depth=1) #entropy or gini
    model.fit(x, y)
    
    print "y_hat=", model.predict(x)
    
    
    
    D = np.mat(np.ones((len(x), 1)) / len(x))
    
    # print buildStump(x, y, D)
    classArr = adaBoostTrainDS(x, y, 9)
    print classArr
    
    print addClassify(x,classArr)     
            
