'''
Created on 2020年10月12日

@author: zjf

LeNet5模型，实现手写数字的识别
'''
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def LeNet5Model():
    model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 1)),
            tf.keras.layers.AveragePooling2D(),
            tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.AveragePooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=120, activation='relu'),
            tf.keras.layers.Dense(units=84, activation='relu'),
            tf.keras.layers.Dense(units=10, activation='softmax')
            ])
    model.summary()
    return model


if __name__ == '__main__':
#     # 下载训练数据 (https://www.python-course.eu/neural_network_mnist.php)
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    
    X_train = train_images.reshape((-1, 28, 28, 1))
    X_train = np.pad(X_train, ((0, 0), (2, 2), (2, 2), (0, 0)))  # 改成32 *32 
    Y_train = np.eye(10)[train_labels.reshape(-1)]  # one-hot
    
    X_test = test_images.reshape((-1, 28, 28, 1))
    X_test = np.pad(X_test, ((0, 0), (2, 2), (2, 2), (0, 0)))  # 改成32 *32 
    Y_test = np.eye(10)[test_labels.reshape(-1)]  # one-hot
    
    print(np.shape(X_train))
    print(np.shape(X_test))
    # 显示 一条数据
    img = X_train[2]
    print(Y_train[2])
    print(np.shape(img))
    plt.imshow(img[:, :, 0], cmap="Greys")
    plt.show()

    # 构建模型并训练
    model = LeNet5Model()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x=X_train, y=Y_train, epochs=100, batch_size=640)
    
    preds = model.evaluate(x=X_train, y=Y_train)
    print("Train_Accuracy : " + str(preds[1] * 100))
    preds = model.evaluate(x=X_test, y=Y_test)
    print("Test_Accuracy : " + str(preds[1] * 100))
        
