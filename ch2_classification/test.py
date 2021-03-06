# coding=utf-8
'''
Created on 2017年5月5日

@author: zjf
'''

from numpy import * 
import pandas as pd
import numpy as np
import scipy.optimize as opt  

def sigmoid(z):  
    return 1 / (1 + np.exp(-z))

def cost(theta, X, y):  
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    return np.sum(first - second) / (len(X))

def gradient(theta, X, y):  
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X * theta.T) - y

    for i in range(parameters):
        term = np.multiply(error, X[:,i])
        grad[i] = np.sum(term) / len(X)

    return grad

def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testdata1.txt')
    for line in fr.readlines():
        lineArr = line.strip().split(',')
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def gradAscent2(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m,n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 5000
    weights = ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose()* error
        print computeCostJ(dataMatIn, classLabels,weights)
    return weights

def computeCostJ(X, Y,theta):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(Y)
    first = np.multiply(-y, np.log(sigmoid(X * theta)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta)))
    return np.sum(first - second) / (len(X))
        
if __name__ == '__main__':
    dataMat,labelMat = loadDataSet()
    print gradAscent2(dataMat,labelMat)
