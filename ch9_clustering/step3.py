# -*- coding:utf-8 -*-
'''
Created on 2017年6月4日

@author: Jeffrey Zhou
'''

'''
DBSCAN算法的引入
'''

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn.datasets as ds
import matplotlib.colors
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    
    #制造1000个数据，4个簇
    N = 1000
    centers = [[1, 2], [-1, -1], [1, -1], [-1, 1]]
    data, y = ds.make_blobs(N, n_features=2, centers=centers, cluster_std=[0.5, 0.25, 0.7, 0.5], random_state=0)
    data = StandardScaler().fit_transform(data)
    
    #可选参数(0.2,5) (0.2,10),(0.2,15)  (0.3,5) (0.3,10),(0.3,15) 
    model = DBSCAN(eps=0.2, min_samples=5)
    model.fit(data)
    y_hat = model.labels_
    
    
    #展示数据
    cm = mpl.colors.ListedColormap(['r', 'g', 'b', 'm', 'c'])
    plt.subplot(211)
    plt.scatter(data[:, 0], data[:, 1],s=2, c=y, cmap=cm)
    plt.title('Prime', fontsize=17)
    plt.grid(b=True, ls=':')
    
    plt.subplot(212)
    plt.scatter(data[:, 0], data[:, 1],s=2, c=y_hat, cmap=cm)
    plt.title('(0.2,5)', fontsize=17)
    plt.grid(b=True, ls=':')
   
    plt.show()