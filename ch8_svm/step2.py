# -*- coding:utf-8 -*-
'''
Created on 2017年5月30日

@author: Jeffrey Zhou
'''

'''
SVM对鸢尾花数据分类
自建SVM 
'''

import pandas as pd
from sklearn import svm
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def selectJrand(i, m):
    j = i
    while(j == i):
        j = int(np.random.uniform(0, m))
    return j
 
def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    x = np.mat(dataMatIn); 
    y = np.mat(classLabels).transpose()
    b = 0
    m, n = np.shape(x)
    alphas = np.mat(np.zeros((m, 1)))
    niter = 0
    while(niter < maxIter):
        alphaPairsChanges = 0
        for i in range(m):
            fXi = float(np.multiply(alphas, y).T * (x * x[i, :].T)) + b
            Ei = fXi - float(y[i])
            if((y[i] * Ei < -toler) and (alphas[i] < C)) or ((y[i] * Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i, m) #随机选出第二个α值
                fXj = float(np.multiply(alphas, y).T * (x * x[j, :].T)) + b
                Ej = fXj - float(y[j])
                alphaIold = alphas[i].copy();
                alphaJold = alphas[j].copy();
                #确保α值在0和C之间
                if (y[i] != y[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H: 
                    #print ('L==H');
                    continue
                eta = 2.0 * x[i, :] * x[j, :].T - x[i, :] * x[i, :].T - x[j, :] * x[j, :].T
                if eta >= 0:
                    #print ('eta >=0');
                    continue
                #修改第i个α的同时相反方便修改第j个α值
                alphas[j] -= y[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    #print ('j not movint enought'); 
                    continue
                alphas[i] += y[j] * y[i] * (alphaJold - alphas[j])
 
                b1 = b - Ei - y[i] * (alphas[i] - alphaIold) * x[i, :] * x[i, :].T - y[j] * (alphas[j] - alphaJold) * x[i, :] * x[j, :].T
                b2 = b - Ej - y[i] * (alphas[i] - alphaIold) * x[i, :] * x[j, :].T - y[j] * (alphas[j] - alphaJold) * x[j, :] * x[j, :].T
                if (0 < alphas[i]) and (C > alphas[i]): b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2
                else : b = (b1 + b2) / 2.0
 
                alphaPairsChanges += 1
                #print ("iter :%d i:%d,paris changed %d" % (niter, i, alphaPairsChanges))
        if (alphaPairsChanges == 0):niter += 1
        else :niter = 0
        #print ("iteration number:%d" % niter)
    return b, alphas
 
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

    