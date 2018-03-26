# -*- coding:utf-8 -*-
'''
Created on 2017年5月30日

@author: Jeffrey Zhou
'''

'''
SVM对鸢尾花数据分类
自建SVM ——增加核函数
'''

import pandas as pd
from sklearn import svm
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    dataMat = [];labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat
 
def selectJrand(i, m):
    j = i
    while(j == i):
        j = int(random.uniform(0, m))
    return j
 
def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj
 
def kernelTrans(X, A, kTup):
    m, n = shape(X)
    K = mat(zeros((m, 1)))
    if kTup[0] == 'lin':K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = exp(K / (-1 * kTup[1] ** 2))
    else: raise NameError('Houston We Hav problem -- That Kernel is not recognized')
    return K
 
 
class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))
        self.K = mat(zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)
 
def calcEk(oS, k):
    fXk = float(multiply(oS.alphas, oS.labelMat).T * oS.K[:, k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek
 
def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i:continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k;maxDeltaE = deltaE;Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej
 
def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]
 
def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or\
     ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:print "L==H"; return 0
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]
        if eta >= 0 :print "eta>=0";return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)
        if (abs(oS.alphas[j] - alphaIold) < 0.00001):
            print "j not moving enough"; return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        updateEk(oS, i)
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, j] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[j, j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif(0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0
 
def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler, kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while(iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
            # print "fullSet iter :%d i:%d,paris changed %d" % (iter, i, alphaPairsChanged)
            iter += 1
        else:
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print "non-bound iter :%d i:%d,paris changed %d" % (iter, i, alphaPairsChanged)
            iter += 1
        if entireSet:
            entireSet = False
        elif(alphaPairsChanged == 0):
            entireSet = True
        print "iteration number %d" % iter
    return oS.b, oS.alphas
 
dataMat, labelMat = loadDataSet('testSet.txt')
b, alphas = smoP(dataMat, labelMat, 0.6, 0.001, 40)
print b, alphas
 
def loadDataSet(fileName):
    dataMat = [];labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

def predict(alphas,b,y,x,dt):
    fXi = float(np.multiply(alphas, y).T * (x * dt.T)) + b
    return fXi

if __name__ == '__main__':
    path = '..\\ch2_classification\\iris.data'  # 数据文件路径
    data = pd.read_csv(path, header=None)
    x, y = data[[0,1,2,3]], data[4]
    y = pd.Categorical(y).codes
    x = x[[0, 1]] #取花萼长度，花萼宽度 两个属性
    
    #为了方便，将数据清先成2类，并把y变成-1和1
    y1 = y.copy()
    y1[y==0] = -1
    y1[y==1] = 1
    y1[y==2] = 1
    y = y1
    
    #x, y = loadDataSet('testSet.txt')
    
    #建模
    mode = svm.SVC(C=0.6,kernel='linear', decision_function_shape='ovr')
    mode.fit(x,y)
    print(mode.dual_coef_, mode.intercept_)
#     test = mode.predict(x[0,:])
#     print(test)
    
# print "iter :%d i:%d,paris changed %d" % (1, 2, 3)
#  
#  
    b, alphas = smoSimple(x, y, 0.6, 0.001, 40)
    print (b, alphas[alphas > 0])
#  
# #计算出支持向量
# shape(alphas[alphas > 0])
# for i in range(100):
#     if alphas[i] > 0:
#         print dataMat[i], labelMat[i]

#     # 画图
#     x1_min, x2_min = x.min()
#     x1_max, x2_max = x.max()
#     x1, x2 = np.mgrid[x1_min:x1_max:500j, x2_min:x2_max:500j]  # 生成网格采样点
#     grid_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点
#     grid_hat = mode.predict(grid_test)       # 预测分类值
#     grid_hat = grid_hat.reshape(x1.shape)  # 使之与输入的形状相同
# 
#     cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
#     cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
#     plt.figure(facecolor='w')
#     plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)
#     plt.scatter(x[0], x[1], c=y, edgecolors='k', s=50, cmap=cm_dark)      # 样本
#     plt.xlim(x1_min, x1_max)
#     plt.ylim(x2_min, x2_max)
#     plt.grid(b=True, ls=':')
    plt.show()

    