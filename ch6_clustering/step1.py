# -*- coding:utf-8 -*-
'''
Created on 2017年5月31日

@author: Jeffrey Zhou
'''
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sklearn.datasets as ds

if __name__ == '__main__':
    N = 400
    centers = 4
    data, y = ds.make_blobs(n_samples=N, n_features=2, centers=centers, random_state=2)
    
    mode = KMeans(n_clusters=4)
    y_hat = mode.fit_predict(data)
    
    plt.subplot(211)
    plt.scatter(data[:, 0], data[:, 1], c=y, s=30)
    plt.grid(True)
    
    plt.subplot(212)
    plt.scatter(data[:, 0], data[:, 1], c=y_hat, s=30)
    plt.grid(True)
    plt.show()