# -*- coding:utf-8 -*-
'''
Created on 2018年8月1日
@author: zjf
'''
'''
构建基本的RNN模型，输入输出数量相同
'''

import numpy as np
from copy import deepcopy
from chi_rnn.step1 import loadData
import random


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def rnn_cell_forward(xt, a_prev, parameters):
    """
            一个cell的向前计算
    """
    Wax, Waa, Wya, ba, by = parameters["Wax"], parameters["Waa"], parameters["Wya"], parameters["ba"], parameters["by"]

    a_next = np.tanh(np.dot(Waa, a_prev) + np.dot(Wax, xt) + ba)
    yt_hat = softmax(np.dot(Wya, a_next) + by) 
    cache = (a_next, a_prev, xt, yt_hat)
    return a_next, yt_hat, cache


def rnn_forward(x, a0, parameters):
    """
            一层的向前计算
    """
    caches = []
    _, T_x = x.shape
    n_y, _ = parameters["Wya"].shape
    y_hat = np.zeros((n_y, n_y, T_x))
    a_next = a0

    for t in range(T_x):
        a_next, yt_hat, cache = rnn_cell_forward(x[:, t].reshape((-1, 1)), a_next, parameters)
        y_hat[:, :, t] = yt_hat
        caches.append(cache)
    return y_hat, caches


def rnn_cell_backward(da_next, dy, cache, parameters, xt):
    """
            一个cell的向后计算
    """
    (a_next, a_prev, _, _) = cache
    
    # 计算梯度
    da = np.dot(parameters['Wya'].T, dy) + da_next  # gradients['da_next']
    dwya = np.dot(dy, a_next.T)
    dby = dy
    dtanh = (1 - a_next * a_next) * da  # backprop through tanh nonlinearity
    dba = np.sum(dtanh, keepdims=True, axis=-1)
    dxt = np.dot(parameters['Wax'].T, dtanh)
    dWax = np.dot(dtanh, xt.T)
    dWaa = np.dot(dtanh, a_prev.T)
    da_prev = np.dot(parameters['Waa'].T, dtanh)
    
    return {"dxt":dxt, "da_prev":da_prev, "dWax":dWax, "dWaa":dWaa, "dba":dba, "dWya":dwya, "dby":dby}


def rnn_backward(parameters, caches, X, Y):
    """
            一层的向前计算
    """
    n_a, _ = parameters["Wax"].shape
    n_x, T_x = X.shape
    dx = np.zeros((n_x, 1, T_x))
    dWax = np.zeros((n_a, n_x))
    dWaa = np.zeros((n_a, n_a))
    dba = np.zeros((n_a, 1))
    da_prevt = np.zeros((n_a, 1))
    dWya = np.zeros((n_x, n_a))  # np.zeros_like(dWax).T
    dby = np.zeros((n_x, 1))
    for t in reversed(range(T_x)):
        (_, _, _, y_hat) = caches[t]
        dy = y_hat - Y[:, t].reshape((-1, 1))
        gradients = rnn_cell_backward(da_prevt, dy, caches[t], parameters, X[:, t].reshape((-1, 1)))
        da_prevt = gradients["da_prev"]
        dx[:, :, t] = gradients["dxt"]
        dWax += gradients["dWax"]
        dWaa += gradients["dWaa"]
        dba += gradients["dba"]
        dWya += gradients["dWya"]
        dby += gradients["dby"]
        
    da0 = da_prevt
    return {"dx": dx, "da0": da0, "dWax": dWax, "dWaa": dWaa, "dba": dba, "dWya":dWya, "dby":dby}


def train(X, Y, num_iterations, hiddencells=50, learning_rate=0.01):
    # 初使化parameters
    np.random.seed(1)
    print(np.shape(X[0]))
    vocab_size = 27
    seq_length = 7
    Wax = np.random.randn(hiddencells, vocab_size) * 0.01  # input to hidden
    Waa = np.random.randn(hiddencells, hiddencells) * 0.01  # hidden to hidden
    Wya = np.random.randn(vocab_size, hiddencells) * 0.01  # hidden to output
    ba = np.zeros((hiddencells, 1))  # hidden bias
    by = np.zeros((vocab_size, 1))  # output bias
    parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "ba": ba, "by": by}
    plst = {}
    initloss = -np.log(1.0 / vocab_size) * seq_length
    a_prev = np.zeros((hiddencells, 1))
    # 循环迭代
    for i in range(num_iterations):
        index = i % len(X)
        _, T_x = X[index].shape
        # 向前计算
        y_hat, caches = rnn_forward(X[index], a_prev, parameters)
        (a_prev, _, _, _) = caches[-1]
        # 向后计算
        gradients = rnn_backward(parameters, caches, X[index], Y[index])
        # 利用梯度更新参数
        parameters['Wax'] += -learning_rate * np.clip(gradients['dWax'], -5, 5)
        parameters['Waa'] += -learning_rate * np.clip(gradients['dWaa'], -5, 5)
        parameters['ba'] += -learning_rate * np.clip(gradients['dba'], -5, 5)
        parameters['Wya'] += -learning_rate * np.clip(gradients['dWya'], -5, 5)
        parameters['by'] += -learning_rate * np.clip(gradients['dby'], -5, 5)
        
        # 计算损失
        loss = 0
        for t in range(T_x):
            loss -= np.log(np.sum(y_hat[:, 0, t] * Y[index][:, t]))
        initloss = initloss * 0.999 + loss * 0.001  # 进行参数平滑
#         
        if i % 2000 == 0:
            print('Iteration: %d, Loss: %f' % (i, initloss) + '')
            plst[i] = deepcopy(parameters)  # .copy()

    return parameters, plst

    
def predict(parameters, char_to_ix, ix_to_char, startstr):
    vocab_size = parameters['by'].shape[0]
    n_a = parameters['Waa'].shape[1]
    
    idx = startstr
    a_prev = np.zeros([n_a, 1])
    indices = [idx]
    
    # 递推式预测后面的词，总长不超过50
    for _ in range(50):
        x = np.zeros([vocab_size, 1])
        x[idx, 0] = 1
        # 向前计算
        a_prev, y, _ = rnn_cell_forward(x, a_prev, parameters)
        
        idx = y.tolist().index(max(y)) 
        indices.append(idx)
        if idx == char_to_ix['\n']:
            break  # 碰到\n 表示结束
        
    indices.append(char_to_ix['\n'])
    txt = ''.join(ix_to_char[ix] for ix in indices)
    return txt[0].upper() + txt[1:]  # 处理首字母大写


def test():
    np.random.seed(1)
    # test rnn_cell_forward
#     xt = np.random.randn(3, 10)
#     a_prev = np.random.randn(5, 10)
#     Waa = np.random.randn(5, 5)
#     Wax = np.random.randn(5, 3)
#     Wya = np.random.randn(2, 5)
#     ba = np.random.randn(5, 1)
#     by = np.random.randn(2, 1)
#     parameters = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": ba, "by": by}
#     a_next, yt_pred, cache = rnn_cell_forward(xt, a_prev, parameters)
#     print("a_next[4] = ", a_next[4])
#     print("a_next.shape = ", a_next.shape)
#     print("yt_pred[1] =", yt_pred[1])
#     print("yt_pred.shape = ", yt_pred.shape)
    
    # test rnn_forward
#     x = np.random.randn(3, 10, 4)
#     a0 = np.random.randn(5, 10)
#     Waa = np.random.randn(5, 5)
#     Wax = np.random.randn(5, 3)
#     Wya = np.random.randn(2, 5)
#     ba = np.random.randn(5, 1)
#     by = np.random.randn(2, 1)
#     parameters = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": ba, "by": by}
#     y_pred, caches = rnn_forward(x, a0, parameters)
#     print("y_pred[1][3] =", y_pred[1][3])
#     print("y_pred.shape = ", y_pred.shape)
#     print("caches[1][1][3] =", caches[1][1][3])
#     print("len(caches) = ", len(caches))
    # test rnn_cell_backward
#     xt = np.random.randn(3, 10)
#     a_prev = np.random.randn(5, 10)
#     Wax = np.random.randn(5, 3)
#     Waa = np.random.randn(5, 5)
#     Wya = np.random.randn(2, 5)
#     ba = np.random.randn(5, 1)
#     by = np.random.randn(2, 1)
#     parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "ba": ba, "by": by}
#     a_next, yt, cache = rnn_cell_forward(xt, a_prev, parameters)
#     da_next = np.random.randn(5, 10)
#     gradients = rnn_cell_backward(da_next, cache, yt, parameters, xt)
#     print("gradients[\"dxt\"][1][2] =", gradients["dxt"][1][2])
#     print("gradients[\"dxt\"].shape =", gradients["dxt"].shape)
#     print("gradients[\"da_prev\"][2][3] =", gradients["da_prev"][2][3])
#     print("gradients[\"da_prev\"].shape =", gradients["da_prev"].shape)
#     print("gradients[\"dWax\"][3][1] =", gradients["dWax"][3][1])
#     print("gradients[\"dWax\"].shape =", gradients["dWax"].shape)
#     print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
#     print("gradients[\"dWaa\"].shape =", gradients["dWaa"].shape)
#     print("gradients[\"dba\"][4] =", gradients["dba"][4])
#     print("gradients[\"dba\"].shape =", gradients["dba"].shape)
    # test rnn_backward
#     x = np.random.randn(3, 10, 4)
#     y = np.random.randn(10)
#     a0 = np.random.randn(5, 10)
#     Wax = np.random.randn(5, 3)
#     Waa = np.random.randn(5, 5)
#     Wya = np.random.randn(2, 5)
#     ba = np.random.randn(5, 1)
#     by = np.random.randn(2, 1)
#     parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "ba": ba, "by": by}
#     y_hat, caches = rnn_forward(x, a0, parameters)
#     da = np.random.randn(5, 10, 4)
#     gradients = rnn_backward(parameters, caches, x, y)
#     print("gradients[\"dx\"][1][2] =", gradients["dx"][1][2])
#     print("gradients[\"dx\"].shape =", gradients["dx"].shape)
#     print("gradients[\"da0\"][2][3] =", gradients["da0"][2][3])
#     print("gradients[\"da0\"].shape =", gradients["da0"].shape)
#     print("gradients[\"dWax\"][3][1] =", gradients["dWax"][3][1])
#     print("gradients[\"dWax\"].shape =", gradients["dWax"].shape)
#     print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
#     print("gradients[\"dWaa\"].shape =", gradients["dWaa"].shape)
#     print("gradients[\"dba\"][4] =", gradients["dba"][4])
#     print("gradients[\"dba\"].shape =", gradients["dba"].shape)


if __name__ == '__main__':
#     test()
    np.set_printoptions(linewidth=300)
    np.random.seed(0)
    
    rows, char_to_ix, ix_to_char, vocab, maxlen = loadData("data/dinos.txt")
  
    # 数据进行one-hot编码
    X = []
    for row in rows:
        dx = np.zeros((27, len(row)))
        for i in range(len(row)):
            if (row[i] != None):
                dx[row[i], i] = 1
        X.append(dx)
    
    # 偏移1位，形成Y
    Y = []
    dtadd = np.zeros((27, 1))
    dtadd[0, 0] = 1
    for dx in X:
        Y.append(np.hstack((dx[:, 1:], dtadd)))
    
    # 训练
    _, plst = train(X, Y, 10000)  # 35000
    
    # 预测
    for item in plst:
        print("\nIteration:", item)
        parameters = plst[item]
        seed = 0
        for name in range(5):
            startstr = random.randint(0, len(vocab) - 1)
            newname = predict(parameters, char_to_ix, ix_to_char, startstr)
            print(newname.strip())
            seed += 1
