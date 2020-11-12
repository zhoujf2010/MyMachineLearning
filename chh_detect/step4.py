'''
Created on 2020年11月12日

@author: zjf
'''

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras import backend
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.keras.layers import VersionAwareLayers
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.utils import layer_utils
import scipy


def VGG19(include_top=True, weights='imagenet', input_tensor=None,
    input_shape=None, pooling=None, classes=1000, classifier_activation='softmax'):
    layers = VersionAwareLayers()
    # Determine proper input shape
    input_shape = imagenet_utils.obtain_input_shape(
        input_shape,
        default_size=224,
        min_size=32,
        data_format=backend.image_data_format(),
        require_flatten=include_top,
        weights=weights)
    
    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    # Block 1
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    
    # Block 2
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    
    # Block 3
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    
    # Block 4
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    
    # Block 5
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    
    if include_top:
        # Classification block
        x = layers.Flatten(name='flatten')(x)
        x = layers.Dense(4096, activation='relu', name='fc1')(x)
        x = layers.Dense(4096, activation='relu', name='fc2')(x)
        imagenet_utils.validate_activation(classifier_activation, weights)
        x = layers.Dense(classes, activation=classifier_activation,
                         name='predictions')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)
    
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = training.Model(inputs, x, name='vgg19')
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


class StyleContentModel(tf.keras.models.Model):

    def __init__(self, style_layers, content_layer, model):
        super(StyleContentModel, self).__init__()
        self.vgg = self.vgg_layers(style_layers + [content_layer], model)
        self.style_layers = style_layers
        self.content_layers = content_layer
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False
    
    def vgg_layers(self, layer_names, model):
        """ Creates a vgg model that returns a list of intermediate output values."""
        model.trainable = False
        outputs = [model.get_layer(name).output for name in layer_names]
        model = tf.keras.Model([model.input], outputs)
        return model
        
    def gram_matrix(self, input_tensor):
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
        return result / (num_locations)

    def call(self, inputs):
        "Expects float input in [0,1]"
        outputs = self.vgg(inputs * 255.0)
        style_outputs, content_outputs = (outputs[:self.num_style_layers], outputs[self.num_style_layers:])
        
        style_outputs = [self.gram_matrix(style_output) for style_output in style_outputs]
        style_dict = {style_name:value for style_name, value in zip(self.style_layers, style_outputs)}
        
        return {'content':content_outputs[0], 'style':style_dict}

    
if __name__ == '__main__':
    # 定义并加载vgg-19模型
    model = VGG19()
    # https://storage.googleapis.com/tensorflow/keras-applications/vgg19/vgg19_weights_tf_dim_ordering_tf_kernels.h5
    model.load_weights("vgg19_weights_tf_dim_ordering_tf_kernels.h5")

    content_image = image_to_tensor("datasets/images/louvre.jpg")
    style_image = image_to_tensor("datasets/images/vangogh.jpg")
    
    mixedimage = tf.Variable(content_image)
    
    # 内容层将提取出我们的 feature maps （特征图）
    content_layer = 'block5_conv2'
    # 我们感兴趣的风格层
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    
    extractor = StyleContentModel(style_layers, content_layer, model)
    opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    style_weight = 1e-2
    content_weight = 1e4
    
    for n in range(2):
        print("range:", n)
        with tf.GradientTape() as tape:
            outputs = extractor(mixedimage)
            style_outputs = outputs['style']
            content_outputs = outputs['content']
            
            style_targets = extractor(style_image)['style']
            content_targets = extractor(content_image)['content']
        
            style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2) 
                                   for name in style_outputs.keys()]) / len(style_layers)
            content_loss = tf.reduce_mean((content_outputs - content_targets) ** 2) 
            loss = style_loss * style_weight + content_loss * content_weight
            
            grad = tape.gradient(loss, mixedimage)
            opt.apply_gradients([(grad, mixedimage)])
            
            tmp = tf.clip_by_value(mixedimage, clip_value_min=0.0, clip_value_max=1.0)
            mixedimage.assign(tmp)
    
    scipy.misc.imsave("result.jpg", tensor_to_image(mixedimage))
    
#     plt.subplot(2, 2, 1)
#     plt.imshow(tensor_to_image(content_image))
#     plt.title("content")
#     
#     plt.subplot(2, 1, 2)
#     plt.imshow(tensor_to_image(style_image))
#     plt.title("style")
#     
#     plt.subplot(2, 2, 2)
#     plt.imshow(tensor_to_image(mixedimage))
#     plt.title("mixed")
#     plt.show()

