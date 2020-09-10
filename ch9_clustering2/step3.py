'''
Created on 2020年9月9日

@author: zjf

普聚类的实现
'''

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans
np.random.seed(1)


def adjMatrix_full(S, sigma=1.0):
    N = len(S)
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            A[i][j] = np.exp(-np.sum((data[i] - data[j]) ** 2) / 2 / sigma / sigma)
            A[j][i] = A[i][j] 
    return A


def adjMatrix_epson(S, epson):
    N = len(S)
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            if np.sum((data[i] - data[j]) ** 2) <= epson:
                A[i][j] = epson
                A[j][i] = A[i][j] 
    return A


def adjMatrix_Knn(data, k, sigma=1.0):
    m = len(data)
    S = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            S[i][j] = 1.0 * np.sum((data[i] - data[j]) ** 2)
            S[j][i] = S[i][j]
            
    A = np.zeros((m, m))
    for i in range(m):
        dist_with_index = zip(S[i], range(m))
        dist_with_index = sorted(dist_with_index, key=lambda x:x[0])
        neighbours_id = [dist_with_index[m][1] for m in range(k + 1)] 
  
        for j in neighbours_id:
            A[i][j] = np.exp(-S[i][j] / 2 / sigma / sigma)
            A[j][i] = A[i][j] 
    return A


if __name__ == '__main__':
    data, y = datasets.make_circles(500, factor=0.5, noise=0.05)  # 造两个圈的数据
#     
    # 传统keams，用于对比
    y_km = KMeans(n_clusters=2).fit_predict(data)
     
    # 普聚类
#     w = adjMatrix_full(data)
#     w = adjMatrix_epson(data,0.3)
    w = adjMatrix_Knn(data, 8)
     
    D = np.diag(np.sum(w, axis=1))  # 计算度矩阵
    L = D - w  # 拉普拉斯矩阵(未正则）

    # 正则(对称）
    sqrtDegreeMatrix = np.diag(1.0 / (np.sum(w, axis=1) ** (0.5)))
    L = np.dot(np.dot(sqrtDegreeMatrix, L), sqrtDegreeMatrix)
    
    # 正则(随机游走对称）
#     sqrtDegreeMatrix = np.diag(1.0 / (np.sum(w, axis=1)))
#     laplacianMatrix = np.dot(sqrtDegreeMatrix, laplacianMatrix)
     
    x, V = np.linalg.eig(L)
    x = zip(x, range(len(x)))
    x = sorted(x, key=lambda x:x[0])
    U = np.vstack([V[:, i] for (v, i) in x[:2]]).T
    y_lps = KMeans(n_clusters=2).fit_predict(U)
    
    plt.subplot(121)
    plt.scatter(data[:, 0], data[:, 1], s=10, c=y_km)
    plt.title("Kmeans Clustering")
    plt.subplot(122)
    plt.scatter(data[:, 0], data[:, 1], s=10, c=y_lps)
    plt.title("Spectral Clustering")
    plt.show()
