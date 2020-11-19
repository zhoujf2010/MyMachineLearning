# -*- coding:utf-8 -*-
'''
Created on 2018年12月22日

@author: zjf
'''
'''
利用TF实现实现生成莎士比亚风格的文章
参考：https://www.tensorflow.org/tutorials/sequences/text_generation
'''

import tensorflow as tf
import numpy as np
import sys


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    out = np.random.choice(range(len(chars)), p=probas.ravel())
    return out


def generate_output(indices_char, usr_input, predict):
    generated = usr_input
    sentence = ('{0:0>' + str(Tx) + '}').format(usr_input).lower()
    sys.stdout.write(generated)
    
    for _ in range(400):
        x_pred = np.zeros((1, Tx, len(chars)))
        for t, char in enumerate(sentence):
            if char != '0':
                x_pred[0, t, char_indices[char]] = 1.

        preds = predict(x_pred)  # 预测下一个字符
        next_index = sample(preds, temperature=1.0)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

        sys.stdout.write(next_char)
        sys.stdout.flush()

        if next_char == '\n':
            continue

        
if __name__ == '__main__':
    text = open("data/shakespeare.txt").read().lower()
    text = text[37:]  # 去掉标题
    print('文本长度:', len(text))
    
    # 读出文本，形成输入输出结果
    X = []
    Y = []
    Tx = 40
    stride = 3
    for i in range(0, len(text) - Tx, stride):
        X.append(text[i: i + Tx])
        Y.append(text[i + Tx])

    print('样本数量:', len(X))
    
    print("x[0]=", X[0])
    print("x[1]=", X[1])
    print("y[0]=", Y[0])

    # 采用one-hot编码
    chars = sorted(list(set(text)))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    
    m = len(X)
    n = len(chars)
    x = np.zeros((m, Tx, n), dtype=np.bool)
    y = np.zeros((m, n), dtype=np.bool)
    for i, sentence in enumerate(X):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[Y[i]]] = 1
     
    # 定义模型   
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(40, 38)),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(128),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(38),
        tf.keras.layers.Activation('softmax')
        ])
     
    model.compile(loss='categorical_crossentropy', optimizer='adam')
#     model = tf.keras.models.load_model('data/model_shakespeare_kiank_350_epoch.h5')
    model.summary()  
    
    # 训练
    model.fit(x, y, batch_size=128, epochs=1)
    
    # 应用
    generate_output(indices_char, "the sunlight, beautiful", lambda x: model.predict(x, verbose=0)[0])
    
