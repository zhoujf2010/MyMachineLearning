# -*- coding:utf-8 -*-
'''
Created on 2017年6月2日

@author: Jeffrey Zhou
'''

'''
层次聚类AGNES
可切换2种类型数据，分别再带上噪声
可切换3种算法，进行比较
'''

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
import sklearn.datasets as ds

if __name__ == '__main__':
    np.random.seed(0)
    N = 400
    
    #4块区域数据
#     data1, y1 = ds.make_blobs(n_samples=N, n_features=2, centers=((-1, 1), (1, 1), (1, -1), (-1, -1)),
#                               cluster_std=(0.1, 0.2, 0.3, 0.4), random_state=0)
    n_clusters = 4
    
    #两个月牙形数据
    data1, y1 = ds.make_moons(n_samples=N, noise=.05)
    n_clusters = 2
    
    #加入噪声
    data1 = np.array(data1)
    n_noise = int(0.1*N)
    r = np.random.rand(n_noise, 2)
    data_min1, data_min2 = np.min(data1, axis=0)
    data_max1, data_max2 = np.max(data1, axis=0)
    r[:, 0] = r[:, 0] * (data_max1-data_min1) + data_min1
    r[:, 1] = r[:, 1] * (data_max2-data_min2) + data_min2
    data1_noise = np.concatenate((data1, r), axis=0)
    y1_noise = np.concatenate((y1, [4]*n_noise))
    
    data = data1_noise
    y=y1_noise
    
    connectivity = kneighbors_graph(data, n_neighbors=7, mode='distance', metric='minkowski', p=2, include_self=True)
    connectivity = 0.5 * (connectivity + connectivity.T)
    ac = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean',
             connectivity=connectivity, linkage="ward")#"ward", "complete", "average"
    ac.fit(data)
    y_hat = ac.labels_
    
    #展示数据
    cm = mpl.colors.ListedColormap(['r', 'g', 'b', 'm', 'c'])
    plt.subplot(211)
    plt.scatter(data[:, 0], data[:, 1],s=2, c=y, cmap=cm)
    plt.title('Prime', fontsize=17)
    plt.grid(b=True, ls=':')
    
    plt.subplot(212)
    plt.scatter(data[:, 0], data[:, 1],s=2, c=y_hat, cmap=cm)
    plt.title('ward', fontsize=17)
    plt.grid(b=True, ls=':')
   
    plt.show()
        
    