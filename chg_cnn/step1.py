# -*- coding:utf-8 -*-
'''
Created on 2019年4月1日

@author: zjf
'''

'''
手势识别
'''

import matplotlib.pyplot as plt
import numpy as np
import h5py
import tensorflow as tf

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
#     print(Y_train[6])
#     plt.show()
    
    # 配置一些运行参数
    train = True  # False
    
    if train:
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=8, kernel_size=(4, 4), data_format="channels_last", padding='same', strides=(1, 1)
                                   , input_shape=(64, 64, 3)),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(pool_size=(8, 8), strides=(8, 8) , data_format="channels_last", padding='same'),
            tf.keras.layers.Conv2D(filters=16, kernel_size=(4, 4), padding='same' , data_format="channels_last", strides=(1, 1)),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4) , data_format="channels_last", padding='same'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(6, activation="softmax"),
            ])
        
        model.summary()
        
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(x=X_train, y=Y_train, epochs=100, batch_size=64)
        model.save("tmpmodelsave.h5")
    else:
        model = tf.keras.models.load_model("tmpmodelsave.h5")
       
    preds = model.evaluate(x=X_train, y=Y_train)
    print("Train_Accuracy : " + str(preds[1] * 100))
    preds = model.evaluate(x=X_test, y=Y_test)
    print("Test_Accuracy : " + str(preds[1] * 100))
    
