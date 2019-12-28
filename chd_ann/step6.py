# -*- coding:utf-8 -*-
'''
Created on 2018年5月10日

@author: Jeffrey Zhou
'''

'''
PyTorch的初验
'''

import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
import matplotlib as mpl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def load_dataset():
    np.random.seed(1)
    train_X, train_Y = sklearn.datasets.make_circles(n_samples=300, noise=.05)
    np.random.seed(2)
    test_X, test_Y = sklearn.datasets.make_circles(n_samples=100, noise=.05)
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    test_X = test_X.T
    test_Y = test_Y.reshape((1, test_Y.shape[0]))
    return train_X, train_Y, test_X, test_Y


def showData(X, Y, predict=None):
    # 显示标注数据
    N, _ = 500, 500  # x,y上切分多细
    t0 = np.linspace(X[0, :].min(), X[0, :].max(), N)
    t1 = np.linspace(X[1, :].min(), X[1, :].max(), N)
    x0, x1 = np.meshgrid(t0, t1)  # 填统到二维数组中
    x_test = np.stack((x0.flat, x1.flat), axis=1)

    if predict != None:
        y_hat = predict(x_test).reshape(x0.shape)
        plt.pcolormesh(x0, x1, y_hat, cmap=mpl.colors.ListedColormap(
            ['#77E0A0', '#FF8080']))

    # 显示原数据
    plt.scatter(X[0, :], X[1, :], c=Y[0], s=40, cmap=plt.cm.Spectral)
    plt.show()
        
        
if __name__ == '__main__':
    # 生成数据
    train_X, train_Y, test_X, test_Y = load_dataset()
    
    train_X = train_X.T
    train_Y = train_Y.T
    test_X = test_X.T
    test_Y = test_Y.T
    
    class Neural_Network(nn.Module):

        def __init__(self,):
            super(Neural_Network, self).__init__()
            self.fc1 = nn.Linear(2, 10)  # 输入维度 * 该层内部结点数
            self.fc2 = nn.Linear(10, 1)  # 该层内部结点数（同上层） * 输出 
            self.fc3 = nn.Sigmoid()
            
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            x = self.fc3(x)
            return x
        
    train_X_t = torch.tensor(train_X, dtype=torch.float32)
    train_Y_t = torch.tensor(train_Y, dtype=torch.float32)
    
    torch.manual_seed(0)
    net = Neural_Network()
    print(net)
    
    learning_rate = 0.1
    num_steps = 10000  # 总迭代次数
    
    criterion = nn.MSELoss()  # 损失函数
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)  # 随机梯度下降
    
    costs =[]
    for i in range(num_steps):
        # in your training loop:
        optimizer.zero_grad()  # zero the gradient buffers
        output = net(train_X_t)
        loss = criterion(output, train_Y_t)
        loss.backward()
        costs.append(loss.detach().numpy())
        if i % 1000 == 0:
            print('loss',loss.detach().numpy())
        optimizer.step()  # Does the update
        
    # 预测，并输出
    out = net(torch.tensor(train_X, dtype=torch.float32))
    y_hat = np.round(out.detach().numpy())
    print("ann accuracy:", np.mean(train_Y == y_hat) * 100, "%")
    
    out = net(torch.tensor(test_X, dtype=torch.float32))
    y_hat = np.round(out.detach().numpy())
    print("test accuracy:", np.mean(test_Y == y_hat) * 100, "%")
    
    showData(train_X.T, train_Y.T, lambda testX: 
             np.round(net(torch.tensor(testX, dtype=torch.float32)).detach().numpy()))
    #显示损失函数曲线
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
