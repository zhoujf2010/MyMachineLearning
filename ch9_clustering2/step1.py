'''
Created on 2020年9月8日

@author: zjf

关于k-means的实现
'''

import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets as ds
from scipy.stats import pearsonr,entropy

def distance(x, y):  # 欧几里得相似度计算方法
    return np.sqrt(np.sum(np.power(a - b, 2) for a, b in zip(x, y)))
     
# def distance(x, y):  # 曼哈顿距离
#     return np.sum(np.abs(a - b) for a, b in zip(x, y))

# def distance(x, y):  # 切比雪夫距离
#     return np.abs(x-y).max()

# def distance(x, y):  # 杰卡德相似度计算
#     res=len(set.intersection(*[set(x),set(y)]))
#     union_cardinality=len(set.union(*[set(x),set(y)]))
#     return res/float(union_cardinality)

# def distance(x, y):  # 余弦相似度
#     return np.sum(a * b for a, b in zip(x, y)) / float(np.linalg.norm(x) * np.linalg.norm(y))

# def distance(x, y):  # 皮尔森相似度
#     return pearsonr(x,y)[0]

# def distance(x, y):  # entropy
#     return entropy(x,y)

# def distance(x, y):  # Hellinger距离
#     return 1/np.sqrt(2)*np.linalg.norm(np.sqrt(x)-np.sqrt(y))


def k_means(data, k):
    m = len(data)
    n = len(data[0])  # data dymation
    cluster_center = np.zeros((k, n))  # 簇中心
    cluster = [-1 for _ in range(m)]  # 分类结果，默认都是-1
    
    # 随机选择簇中心
    selected = set()
    for i in range(0, k):
        j = np.random.randint(0, m - 1)
        if j in selected:
            continue
        selected.add(j)
        cluster_center[i] = data[j]
    
    for _ in range(40):  # 主计算，迭代40次
        for i in range(m):  # 所有点寻找最近的中心
            dist = []
            for j in range(k):
                dist.append(distance(data[i], cluster_center[j]))
            cluster[i] = dist.index(np.min(dist))
        
        # 计算新的中心点
        cc_sum = np.zeros((k, n))
        cc_num = np.zeros(k)
        for i in range(m):
            index = cluster[i]
            cc_num[index] += 1
            cc_sum[index] += data[i]
        for i in range(k):
            cluster_center[i] = cc_sum[i] / cc_num[i]
    
    return cluster


if __name__ == '__main__':
    N = 400
    centers = 4
    data, y = ds.make_blobs(n_samples=N, n_features=2, centers=centers, random_state=2)
    
    y_hat = k_means(data, 4)
    
    plt.subplot(211)
    plt.scatter(data[:, 0], data[:, 1], c=y, s=30)
    plt.grid(True)
    
    plt.subplot(212)
    plt.scatter(data[:, 0], data[:, 1], c=y_hat, s=30)
    plt.grid(True)
    plt.show()
    
