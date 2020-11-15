'''
Created on 2020年11月12日

@author: zjf
'''

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import scipy
from operator import itemgetter


def VGG19(include_top=True, input_shape=None, pooling=None, classes=1000, classifier_activation='softmax'):
    # Determine proper input shape
    img_input = tf.keras.layers.Input(shape=input_shape)
    
    # Block 1
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    
    # Block 2
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    
    # Block 3
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    
    # Block 4
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    
    # Block 5
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    
    if include_top:
        # Classification block
        x = tf.keras.layers.Flatten(name='flatten')(x)
        x = tf.keras.layers.Dense(4096, activation='relu', name='fc1')(x)
        x = tf.keras.layers.Dense(4096, activation='relu', name='fc2')(x)
        x = tf.keras.layers.Dense(classes, activation=classifier_activation, name='predictions')(x)
    else:
        if pooling == 'avg':
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = tf.keras.layers.GlobalMaxPooling2D()(x)
    
    # Create model.
    model = tf.keras.Model(img_input, x, name='vgg19')
    return model


def image_to_tensor(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim
    
    new_shape = tf.cast(shape * scale, tf.int32)
    
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return tensor


# 截取中间结果的模型
class StyleContentModel(tf.keras.models.Model):

    def __init__(self, selectedlayers, model):
        super(StyleContentModel, self).__init__()
        self.vgg = self.vgg_layers(selectedlayers, model)
        self.selectedlayers = selectedlayers
    
    def vgg_layers(self, layer_names, model):
        """ Creates a vgg model that returns a list of intermediate output values."""
        model.trainable = False
        outputs = [model.get_layer(name).output for name in layer_names]
        model = tf.keras.Model([model.input], outputs)
        model.trainable = False
        return model

    def call(self, inputs):
        "Expects float input in [0,1]"
        outputs = self.vgg(inputs * 255.0)
        style_dict = {style_name:value for style_name, value in zip(self.selectedlayers, outputs)}
        return style_dict
 
        
def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / (num_locations)   


def total_loss(style_outputs, style_targets, content_outputs, content_targets):
    style_weight = 1e-2  # style权重
    content_weight = 1e4  # 内容权重
    
    style_loss = tf.add_n([tf.reduce_mean((gram_matrix(so) - gram_matrix(st)) ** 2) 
                                   for so, st in zip(style_outputs, style_targets)]) / len(style_layers)

    content_loss = tf.reduce_mean((content_outputs - content_targets) ** 2) 
    loss = style_loss * style_weight + content_loss * content_weight
    return loss

            
if __name__ == '__main__':
    # 定义并加载vgg-19模型
    model = VGG19(input_shape=(224, 224, 3))
    # https://storage.googleapis.com/tensorflow/keras-applications/vgg19/vgg19_weights_tf_dim_ordering_tf_kernels.h5
    model.load_weights("vgg19_weights_tf_dim_ordering_tf_kernels.h5")

    content_image = image_to_tensor("datasets/images/louvre.jpg")
    style_image = image_to_tensor("datasets/images/vangogh.jpg")
    
    mixedimage = tf.Variable(content_image)
    
    # 内容层将提取出我们的 feature maps （特征图）
    content_layer = 'block5_conv2'
    # 我们感兴趣的风格层
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    
    extractor = StyleContentModel(style_layers + [content_layer], model)
    style_targets = list(itemgetter(*style_layers)(extractor(style_image)))
    content_targets = extractor(content_image)[content_layer]
    
    opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
     
    for n in range(4):
        print("range:", n)
        with tf.GradientTape() as tape:
            outputs = extractor(mixedimage)
            style_outputs = list(itemgetter(*style_layers)(outputs))
            content_outputs = outputs[content_layer]

            loss = total_loss(style_outputs, style_targets, content_outputs, content_targets)
             
            grad = tape.gradient(loss, mixedimage)
            opt.apply_gradients([(grad, mixedimage)])
             
            tmp = tf.clip_by_value(mixedimage, clip_value_min=0.0, clip_value_max=1.0)
            mixedimage.assign(tmp)
            
#         if n % 10 == 0:
#             scipy.misc.imsave("result%d.jpg" % n, tensor_to_image(mixedimage))
     
#     scipy.misc.imsave("result.jpg", tensor_to_image(mixedimage))
    
#     mixedimage = image_to_tensor("result90.jpg")
    plt.subplot(2, 2, 1)
    plt.imshow(tensor_to_image(content_image))
    plt.title("content")
     
    plt.subplot(2, 1, 2)
    plt.imshow(tensor_to_image(mixedimage))
    plt.title("mixed")
     
    plt.subplot(2, 2, 2)
    plt.imshow(tensor_to_image(style_image))
    plt.title("style")
    plt.show()

