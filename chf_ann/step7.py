# -*- coding:utf-8 -*-
'''
Created on 2018年5月10日

@author: Jeffrey Zhou
'''

'''
PaddlePaddle的初验
'''

import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
import matplotlib as mpl
import paddle # 导入paddle模块
import paddle.fluid as fluid


def load_dataset():
    np.random.seed(1)
    train_X, train_Y = sklearn.datasets.make_circles(n_samples=300, noise=.05)
    np.random.seed(2)
    test_X, test_Y = sklearn.datasets.make_circles(n_samples=100, noise=.05)
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    test_X = test_X.T
    test_Y = test_Y.reshape((1, test_Y.shape[0]))
    train_X = train_X.astype(np.float32)
    return train_X, train_Y, test_X, test_Y


def showData(X, Y, predict=None):
    # 显示标注数据
    N, M = 500, 500  # x,y上切分多细
    t0 = np.linspace(X[0, :].min(), X[0, :].max(), N)
    t1 = np.linspace(X[1, :].min(), X[1, :].max(), N)
    x0, x1 = np.meshgrid(t0, t1)  # 填统到二维数组中
    x_test = np.stack((x0.flat, x1.flat), axis=1)
    x_test = x_test.astype(np.float32)

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
    print(type(train_X[0]))
    print(np.shape(train_X), np.shape(train_Y))
#     showData(train_X, train_Y)
    
    # 标签层，名称为label,对应输入图片的类别标签
    x = fluid.data(name='x', shape=[None, 2], dtype='float32')
    y = fluid.data(name='y', shape=[None, 1], dtype='int64')
    
    hidden = fluid.layers.fc(input=x, size=4, act='relu')
#     hidden = fluid.layers.fc(input=hidden, size=4, act='relu')
    predict = fluid.layers.fc(input=hidden, size=2, act='sigmoid')

    # 使用类交叉熵函数计算predict和label之间的损失函数
    cost = fluid.layers.cross_entropy(input=predict, label=y)
    # 计算平均损失
    avg_loss = fluid.layers.mean(cost)
    # 计算分类准确率
    acc = fluid.layers.accuracy(input=predict, label=y)
    # 选择Adam优化器
    optimizer = fluid.optimizer.Adam(learning_rate=0.001)
    optimizer.minimize(avg_loss)

    use_cuda = False # 如想使用GPU，请设置为 True
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    lists = []
    step = 0
    for i in range(2000):
        metrics = exe.run(feed={'x':train_X,'y':train_Y},
                          fetch_list=[avg_loss, acc])
        if step % 100 == 0: #每训练100次 打印一次log
            print("Pass %d, Batch %d, Cost %f" % (step, 1, metrics[0]))
        step += 1
        
#     fluid.io.save_params(executor=exe,dirname="./savemodel",main_program=None)
# 
#     fluid.io.load_params(executor=exe, dirname="./savemodel",
#                      main_program=None)
    
    results = exe.run(
        feed={'x':train_X,'y':np.zeros((len(train_X),1),dtype='int64')},
        fetch_list=[predict])
    lab = np.argsort(results)
#     print(results)
#     print(lab[0][:,0])
    
    showData(train_X.T, train_Y.T, lambda testX: 
             np.argsort(exe.run(feed={'x':testX,'y':np.zeros((len(testX),1),dtype='int64')},
                                fetch_list=[predict]))[0][:,0])
   
