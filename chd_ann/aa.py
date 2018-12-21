# -*- coding:utf-8 -*-
'''
Created on 2018年5月10日

@author: Jeffrey Zhou
'''

'''
gputest
'''

import numpy as np
# import pycuda.driver as cuda
# import pycuda.autoinit,pycuda.compiler
import tensorflow as tf
import time
import os
import warnings

# if __name__ == '__main__':
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

A = np.array([[1, 1,1],
              [0, 1,1]])
B = np.array([[2, 0],
              [3, 4],
              [3, 4]])
print("A dot B=", np.dot(A, B))

C = np.zeros((A.shape[0], B.shape[1]))
for i in range(C.shape[0]):
    for j in range(C.shape[1]):
        for p in range(A.shape[1]):
            C[i, j] = C[i, j] + A[i, p]*B[p, j]

print("A dot B 2=", C)

a = tf.Variable(A)
b = tf.Variable(B)
c = tf.matmul(a, b)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("A dot B 3=", sess.run(c))


t1 = time.time()
A = np.random.randn(1000, 10000)
B = np.random.randn(10000, 1000)
C = np.dot(A, B)
t2 = time.time()
print('totally cost1', t2-t1)


t1 = time.time()
a = tf.Variable(tf.random_normal([1000, 10000]))
b = tf.Variable(tf.random_normal([10000, 1000]))
c = tf.matmul(a, b)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(c)

t2 = time.time()
print('totally cost2', t2-t1)

# t1 = time.time()
# A = np.random.randn(1000, 10000)
# B = np.random.randn(10000, 1000)
# C = np.zeros((A.shape[0], B.shape[1]))
# for i in range(C.shape[0]):
#     for j in range(C.shape[1]):
#         for p in range(A.shape[1]):
#             C[i, j] = C[i, j] + A[i, p]*B[p, j]
# t2 = time.time()
# print('totally cost3', t2-t1)t1 = time.time()
# A = np.random.randn(1000, 10000)
# B = np.random.randn(10000, 1000)
# C = np.zeros((A.shape[0], B.shape[1]))
# for i in range(C.shape[0]):
#     for j in range(C.shape[1]):
#         for p in range(A.shape[1]):
#             C[i, j] = C[i, j] + A[i, p]*B[p, j]
# t2 = time.time()
# print('totally cost3', t2-t1)



# a  = np.random.randn(4,4)
# print(a)
# with tf.device('/cpu:0'):
#     a = tf.constant([1.0, 2.0, 3.0], shape=[3], name='a')
#     b = tf.constant([1.0, 2.0, 3.0], shape=[3], name='b')
# with tf.device('/gpu:1'):
#     c = a+b

# #注意：allow_soft_placement=True表明：计算设备可自行选择，如果没有这个参数，会报错。
# #因为不是所有的操作都可以被放在GPU上，如果强行将无法放在GPU上的操作指定到GPU上，将会报错。
# sess = tf.Session(config=tf.ConfigProto(
#     allow_soft_placement=True, log_device_placement=True))
# #sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# sess.run(tf.global_variables_initializer())
# print(sess.run(c))

# t1 = time.time()
# a = tf.Variable(tf.random_normal([1000, 100000]))
# b = tf.Variable(tf.random_normal([100000, 1000]))
# c = tf.matmul(a,b)
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# sess.run(c)
# t2 = time.time()
# print('totally cost', t2-t1)

# t1 = time.time()
# a = np.random.randn(1000, 100000)
# b = np.random.randn(100000, 1000)
# c = np.dot(a,b)
# t2 = time.time()
# print('totally cost', t2-t1)


# from tensorflow.python.client import device_lib as _device_lib
# local_device_protos = _device_lib.list_local_devices()
# print([x.name for x in local_device_protos])
