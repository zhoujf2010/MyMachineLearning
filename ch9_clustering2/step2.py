'''
Created on 2020年9月9日

@author: zjf

canopy实现
'''
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets as ds


def distance(x, y):  # 欧几里得相似度计算方法
    return np.sqrt(np.sum(np.power(a - b, 2) for a, b in zip(x, y)))
     
def canopy(dataset, r1, r2):
    m = len(dataset)
    canopies = []  # 用于存放最终归类的结果
    cluster = [-1 for _ in range(m)]
    
    while True:
        rand_index = np.random.randint(len(dataset))
        current_center = dataset[rand_index] 
        current_center_list = [] 
        dataset = np.delete(dataset, rand_index, 0)
        dellst = []
        
        for j in range(len(dataset)):
            datum = dataset[j]
            dj = distance(current_center, datum)
            if dj < r1:
                current_center_list = [j]
                dellst.append(datum)
            elif dj < r2:
                current_center_list.append(j)
        canopies.append((current_center, current_center_list)) #统一删除
        dataset = np.delete(dataset, dellst, 0)
        if len(dellst) ==0 and len(current_center_list)==0:
            break
    
    for i,row in enumerate(canopies):
        for item in row[1]:
            cluster[item] = i
    return cluster

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

if __name__ == '__main__':
    N = 400
    centers = 4
    data, y = ds.make_blobs(n_samples=N, n_features=2, centers=centers, random_state=2)
    data = normalization(data)
    
    y_hat = canopy(data, 0.2, 0.6)
    
    plt.subplot(211)
    plt.scatter(data[:, 0], data[:, 1], c=y, s=30)
    plt.grid(True)
    
    plt.subplot(212)
    plt.scatter(data[:, 0], data[:, 1], c=y_hat, s=30)
    plt.grid(True)
    plt.show()
