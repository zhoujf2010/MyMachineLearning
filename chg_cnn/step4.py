'''
Created on 2020年10月13日

@author: zjf

用AlexNet模型，进行图片10分类识别
'''

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def process_images(image, label):
    # Normalize images to have a mean of 0 and standard deviation of 1
    image = tf.image.per_image_standardization(image)
    # Resize images from 32x32 to 277x277
    image = tf.image.resize(image, (227, 227))
    return image, label


def AlexNetModel():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu', input_shape=(227, 227, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(filters=384, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.summary()
    return model


if __name__ == '__main__':
    # 数据集源自：https://www.cs.toronto.edu/~kriz/cifar.html
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    print(np.shape(train_images))
    print(np.shape(test_images))
    
    train_labels = np.eye(10)[train_labels.reshape(-1)]  # one-hot
    
    #图片数据是32*32较小，需要转成227*227的图片，直接转占用内存较大，故可以用tf中的map来处理
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_ds = train_ds.map(process_images).batch(batch_size=64, drop_remainder=True)
    
#     img = train_images[1]
#     print(train_labels[1])
#     print(np.shape(img))
#     plt.imshow(img)
#     plt.show()
    
    # 构建模型并训练
    model = AlexNetModel()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_ds, epochs=100)
    
#     preds = model.evaluate(x=X_train, y=Y_train)
#     print("Train_Accuracy : " + str(preds[1] * 100))
#     preds = model.evaluate(x=X_test, y=Y_test)
#     print("Test_Accuracy : " + str(preds[1] * 100))
    
