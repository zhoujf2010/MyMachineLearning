'''
Created on 2020年11月6日

人脸识别的实现

@author: zjf
'''

import tensorflow as tf
from inception_blocks_v2 import *
import cv2

def triplet_loss(y_true, y_pred, alpha = 0.2):
    """
    Implementation of the triplet loss as defined by formula (3)

    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)

    Returns:
    loss -- real number, value of the loss
    """

    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    ### START CODE HERE ### (≈ 4 lines)
    # Step 1: Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)))
    # Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)))
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = tf.add(tf.subtract(pos_dist,neg_dist), alpha)
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.))
    ### END CODE HERE ###
    return loss


def img_to_encoding(image_path, model):
    img1 = cv2.imread(image_path, 1)
    img = img1[...,::-1]
    img = np.around(np.transpose(img, (0,1,2))/255.0, decimals=12)
    x_train = np.array([img])
    embedding = model.predict_on_batch(x_train)
    return embedding

def verify(image_path, identity, database, model):
    """
    Function that verifies if the person on the "image_path" image is "identity".

    Arguments:
    image_path -- path to an image
    identity -- string, name of the person you'd like to verify the identity. Has to be a resident of the Happy house.
    database -- python dictionary mapping names of allowed people's names (strings) to their encodings (vectors).
    model -- your Inception model instance in Keras

    Returns:
    dist -- distance between the image_path and the image of "identity" in the database.
    door_open -- True, if the door should open. False otherwise.
    """

    ### START CODE HERE ###

    # Step 1: Compute the encoding for the image. Use img_to_encoding() see example above. (≈ 1 line)
    encoding = img_to_encoding(image_path, model)

    # Step 2: Compute distance with identity's image (≈ 1 line)
    dist = np.linalg.norm(encoding - database[identity])

    # Step 3: Open the door if dist < 0.7, else don't open (≈ 3 lines)
    if dist < 0.7:
        print("It's " + str(identity) + ", welcome home!")
        door_open = True
    else:
        print("It's not " + str(identity) + ", please go away")
        door_open = False

    ### END CODE HERE ###

    return dist, door_open

if __name__ == '__main__':

    tf.random.set_seed(1)
    y_true = (None, None, None)
    y_pred = (tf.random.normal([3, 128], mean=6, stddev=0.1, seed = 1),
              tf.random.normal([3, 128], mean=2, stddev=3.5, seed = 1),
              tf.random.normal([3, 128], mean=3, stddev=4, seed = 1))
    loss = triplet_loss(y_true, y_pred)
#     
    print("loss = " + str(loss.numpy()))
   
    FRmodel = faceRecoModel(input_shape=(96, 96,3))
    print("Total Params:", FRmodel.count_params())
    
    FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
#     load_weights_from_FaceNet(FRmodel)
    FRmodel.load_weights("modelsave/model")
    
#     FRmodel.save_weights("modelsave/model")
    
    database = {}
    database["danielle"] = img_to_encoding("datasets/images/danielle.png", FRmodel)
    database["younes"] = img_to_encoding("datasets/images/younes.jpg", FRmodel)
    database["tian"] = img_to_encoding("datasets/images/tian.jpg", FRmodel)
    database["andrew"] = img_to_encoding("datasets/images/andrew.jpg", FRmodel)
    database["kian"] = img_to_encoding("datasets/images/kian.jpg", FRmodel)
    database["dan"] = img_to_encoding("datasets/images/dan.jpg", FRmodel)
    database["sebastiano"] = img_to_encoding("datasets/images/sebastiano.jpg", FRmodel)
    database["bertrand"] = img_to_encoding("datasets/images/bertrand.jpg", FRmodel)
    database["kevin"] = img_to_encoding("datasets/images/kevin.jpg", FRmodel)
    database["felix"] = img_to_encoding("datasets/images/felix.jpg", FRmodel)
    database["benoit"] = img_to_encoding("datasets/images/benoit.jpg", FRmodel)
    database["arnaud"] = img_to_encoding("datasets/images/arnaud.jpg", FRmodel)

    verify("datasets/images/camera_0.jpg", "younes", database, FRmodel)
    
    verify("datasets/images/camera_2.jpg", "kian", database, FRmodel)
    
    
        