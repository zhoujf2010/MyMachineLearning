# -*- coding:utf-8 -*-
'''
Created on 2019年4月1日

@author: zjf
'''

'''
手动实现CNN相关函数
'''

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib
from skimage import data, filters, img_as_ubyte

matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


def zero_pad(X, pad):
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
    return X_pad


def Test_zero_pad():
    x = np.random.randn(4, 3, 3, 2)
    x_pad = zero_pad(x, 2)
    print ("x.shape =", x.shape)
    print ("x_pad.shape =", x_pad.shape)
    print ("x[1,1] =", x[1, 1])
    print ("x_pad[1,1] =", x_pad[1, 1])
    
    _, axarr = plt.subplots(1, 2)
    axarr[0].set_title('x')
    axarr[0].imshow(x[0, :, :, 0])
    axarr[1].set_title('x_pad')
    axarr[1].imshow(x_pad[0, :, :, 0])
    plt.show()


def conv_step(X, sfilter, stride=1):
    _, f = np.shape(sfilter)
    height, width = np.shape(X)
    n_H = (height - f) // stride + 1
    n_W = (width - f) // stride + 1
    filtered = np.zeros((n_H, n_W))
    for y in range(height - 2):
        for x in range(width - 2):
            vert_start = stride * y
            vert_end = vert_start + f
            horiz_start = stride * x
            horiz_end = horiz_start + f

            a_slice = X[vert_start:vert_end, horiz_start:horiz_end]
            s = a_slice * sfilter
            filtered[y, x] = np.sum(s)
    return filtered
    

def Test_conv_single_step():
    a_slice_prev = np.random.randn(3, 3)
    W = np.random.randn(3, 3)
    b = np.random.randn(1, 1, 1)
    
    Z = conv_step(a_slice_prev, W)
    print("Z =", Z)
    
    pic = Image.open('datasets\\pic.png')
    pic = pic.convert('L')
    data = np.array(pic).astype(np.float) / 255
    print(np.shape(data))
    
    w = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    data = conv_step(data, w, 1)
    print(np.shape(data))
    data = np.pad(data, ((1, 1), (1, 1)))
    print(np.shape(data))

    plt.figure(figsize=(10, 5), facecolor='w')
    plt.subplot(121)
    plt.imshow(pic, cmap=plt.cm.gray, interpolation='nearest')
    plt.title(u'原始图片', fontsize=18)
    plt.subplot(122)
    plt.imshow(data, cmap=plt.cm.gray, interpolation='nearest')
    plt.title(u'处理后图片', fontsize=18)
    plt.show()

    
def pooling(X, f, stride, mode="max"):
    n_H_prev, n_W_prev = np.shape(X)
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    
    A = np.zeros((n_H, n_W))
    for h in range(n_H):
        for w in range(n_W):
            vert_start = h * stride
            vert_end = vert_start + f
            horiz_start = w * stride
            horiz_end = horiz_start + f
            a_prev_slice = X[vert_start:vert_end, horiz_start:horiz_end]
            
            if mode == "max":
                A[h, w] = np.max(a_prev_slice)
            elif mode == "average":
                A[ h, w] = np.mean(a_prev_slice)
    return A

    
def Test_pooling():
    A_prev = np.random.randn(4, 4)
    
    A = pooling(A_prev, 2, 2)
    print("mode = max")
    print("A =", A)
    print()
    A = pooling(A_prev, 2, 2, mode="average")
    print("mode = average")
    print("A =", A)


if __name__ == '__main__':
    np.random.seed(1)
#     Test_zero_pad();
#     Test_conv_single_step()
    Test_pooling()
