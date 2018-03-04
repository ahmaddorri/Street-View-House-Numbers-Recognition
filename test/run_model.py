__author__ = 'ahmaddorri'

import tensorflow as tf
import matplotlib.image as img
import numpy as np
import cv2

a = "images/outfile1.jpg"
image = img.imread(a)
data = []
data.append(image)
data=np.array(data)

import pickle


def im2gray(image):
    '''Normalize images'''
    image = image.astype(float)
    # Use the Conversion Method in This Paper:
    # [http://www.eyemaginary.com/Rendering/TurnColorsGray.pdf]
    image_gray = np.dot(image, [[0.2989],[0.5870],[0.1140]])
    return image_gray

test_img = im2gray(data)[:,:,:,0]



def GCN(image, min_divisor=1e-4):
    """Global Contrast Normalization"""

    imsize = image.shape[0]
    mean = np.mean(image, axis=(1,2), dtype=float)
    std = np.std(image, axis=(1,2), dtype=float, ddof=1)
    std[std < min_divisor] = 1.
    image_GCN = np.zeros(image.shape, dtype=float)

    for i in np.arange(imsize):
        image_GCN[i,:,:] = (image[i,:,:] - mean[i]) / std[i]

    return image_GCN

test_img = GCN(test_img)


image_size = 32
num_labels = 11 # 0-9, + blank
num_channels = 1 # grayscale

def reformat(dataset):
    dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
    return dataset


test_dataset = reformat(test_img)


# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
    saver = tf.train.import_meta_graph('net3.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    graph = tf.get_default_graph()
    x_image = graph.get_tensor_by_name("x_image:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")
    y_conv = graph.get_tensor_by_name("y_conv:0")
    print("Model restored.")
    result = tf.argmax(y_conv,1)
    print(result.eval(feed_dict={x_image: test_dataset, keep_prob: 1.0}))
    conv =y_conv.eval(feed_dict={x_image: test_dataset, keep_prob: 1.0})
    print(conv)