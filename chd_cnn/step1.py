# -*- coding:utf-8 -*-
'''
Created on 2019年4月1日

@author: zjf
'''

import matplotlib.pyplot as plt
import math
import numpy as np
import h5py
import tensorflow as tf


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    
    m = X.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[permutation, :]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :, :, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m, :, :, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

if __name__ == '__main__':
    # load data
    train_dataset = h5py.File('datasets/train_signs.h5', "r")
    X_train = np.array(train_dataset["train_set_x"][:])
    X_train = X_train / 255.  # change to 0~1
    Y_train = np.array(train_dataset["train_set_y"][:])
    Y_train = Y_train.reshape((1, Y_train.shape[0]))
    Y_train = np.eye(6)[Y_train.reshape(-1)]  # one-hot
    m = X_train.shape[0]
    
    test_dataset = h5py.File('datasets/test_signs.h5', "r")
    X_test = np.array(test_dataset["test_set_x"][:])
    X_test = X_test / 255.  # change to 0~1
    Y_test = np.array(test_dataset["test_set_y"][:]) 
    Y_test = Y_test.reshape((1, Y_test.shape[0]))
    Y_test = np.eye(6)[Y_test.reshape(-1)]  # one-hot
    classes = np.array(test_dataset["list_classes"][:]) 
    
    print ("X_train shape: " + str(X_train.shape))
    print ("Y_train shape: " + str(Y_train.shape))
    print ("X_test shape: " + str(X_test.shape))
    print ("Y_test shape: " + str(Y_test.shape))
    print ("train num: " + str(m))
    # test显示
#     plt.imshow(X_train[6])
#     print(Y_train[:,6])
#     plt.show()
    
    tf.reset_default_graph()
    tf.set_random_seed(1)  # 为方便观察结果
    
    # 配置一些运行参数
    learning_rate = 0.009
    num_epochs = 100
    minibatch_size = 64
    seed = 3
    costs = []
    
    # 设置占位符
    X = tf.placeholder(tf.float32, [None, 64, 64, 3], "X")
    Y = tf.placeholder(tf.float32, [None, 6], "Y")

    # 定义参数
    W1 = tf.get_variable("W1", [4, 4, 3, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable("W2", [2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    
    # 设置前馈网络
    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding="SAME")
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding="SAME")
    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding="SAME")
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding="SAME")
    P2 = tf.contrib.layers.flatten(P2)
    Z3 = tf.contrib.layers.fully_connected(P2, 6, activation_fn=None)
    
    # 计算成本函数
    costJ = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))
    
    # 定义优化器
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(costJ)
    
    
    # 启动session 计算TF图
    with tf.Session() as sess:
        np.random.seed(1)
        tf.set_random_seed(1)
        # 初使化
        init = tf.global_variables_initializer()
        sess.run(init)
        print("W1=", W1.eval()[1, 1, 1])  # 测试一下输出值
        print("W2=", W2.eval()[1, 1, 1])
        a = sess.run(Z3, {X: np.random.randn(2,64,64,3), Y: np.random.randn(2,6)})
        print("Z3 = " + str(a))
        a = sess.run(costJ, {X: np.random.randn(4,64,64,3), Y: np.random.randn(4,6)})
        print("cost = " + str(a))
    
        for epoch in range(num_epochs):  # 循环迭代次数
            batchcost = 0
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            # 分批计算
            for batch in minibatches:
                (batch_X, batch_Y) = batch
                _, cost = sess.run([optimizer, costJ], feed_dict={X:batch_X, Y:batch_Y})
                batchcost += cost / int(m / minibatch_size)
             
            if epoch % 5 == 0:
                print ("Cost after epoch %i: %f" % (epoch, batchcost))
            costs.append(batchcost)   
    
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
 
    # Calculate the correct predictions
    predict_op = tf.argmax(Z3, 1)
    correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
 
    # Calculate accuracy on the test set
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(accuracy)
    train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
    test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
    print("Train Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)
        
            