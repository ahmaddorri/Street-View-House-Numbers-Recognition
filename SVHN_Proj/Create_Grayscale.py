__author__ = 'ahmaddorri'


import scipy.io
import numpy as np
import pickle
import tensorflow as tf

import gc
gc.enable()


# load train/test dataset
train_data = scipy.io.loadmat('/Users/ahmaddorri/Desktop/train_32x32.mat', variable_names='X').get('X')
train_labels = scipy.io.loadmat('/Users/ahmaddorri/Desktop/train_32x32.mat', variable_names='y').get('y')
test_data = scipy.io.loadmat('/Users/ahmaddorri/Desktop/test_32x32.mat', variable_names='X').get('X')
test_labels = scipy.io.loadmat('/Users/ahmaddorri/Desktop/test_32x32.mat', variable_names='y').get('y')
#extra_data = scipy.io.loadmat('/Users/ahmaddorri/Downloads/extra_32x32.mat', variable_names='X').get('X')
#extra_labels = scipy.io.loadmat('/Users/ahmaddorri/Downloads/extra_32x32.mat', variable_names='y').get('y')

print(train_data.shape, train_labels.shape)
print(test_data.shape, test_labels.shape)


train_dataset = train_data.transpose((3,0,1,2))
train_label = train_labels[:,0]
test_dataset = test_data.transpose((3,0,1,2))
test_label = test_labels[:,0]
#extra_dataset = extra_data.transpose((3,0,1,2))
#extra_label = extra_labels[:,0]

del train_data
del test_data
del train_labels
del test_labels


#print(train_dataset.shape, train_label.shape)
#print(test_dataset.shape, test_label.shape)
#print(extra_dataset.shape, extra_label.shape)

# create a validation dataset, 4/5 of data from train dataset , 1/5 from extra dataset
#....


def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,:,:]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels
train_dataset, train_label = randomize(train_dataset, train_label)
test_dataset, test_label = randomize(test_dataset, test_label)
#valid_dataset, valid_label = randomize(valid_dataset, valid_label)


import matplotlib.pyplot as plt
import random

def disp_sample_dataset(dataset, label):
    items = random.sample(range(dataset.shape[0]), 8)
    for i, item in enumerate(items):
        plt.subplot(2, 4, i+1)
        plt.axis('off')
        plt.title(label[i])
        plt.imshow(dataset[i])
    plt.show()
#disp_sample_dataset(train_dataset, train_label)


#############################################################################################

#Grayscale

image_size = 32  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

def im2gray(image):
    '''Normalize images'''
    image = image.astype(float)
    # Use the Conversion Method in This Paper:
    # [http://www.eyemaginary.com/Rendering/TurnColorsGray.pdf]
    image_gray = np.dot(image, [[0.2989],[0.5870],[0.1140]])
    return image_gray

train_data_c = im2gray(train_dataset[0:1000])[:,:,:,0]
test_data_c = im2gray(test_dataset[0:1000])[:,:,:,0]

for i in range(1,73):
    train_data_c=np.append(train_data_c,im2gray(train_dataset[1000*i:1000*(i+1)])[:,:,:,0],axis=0)

for i in range(1,26):
    test_data_c=np.append(test_data_c,im2gray(test_dataset[1000*i:1000*(i+1)])[:,:,:,0],axis=0)

train_data_c=np.append(train_data_c,im2gray(train_dataset[73000:])[:,:,:,0],axis=0)
test_data_c=np.append(test_data_c,im2gray(test_dataset[26000:])[:,:,:,0],axis=0)

del train_dataset
del test_dataset

#valid_data_c = im2gray(valid_dataset)[:,:,:,0]

print(train_data_c.shape, train_label.shape)
print(test_data_c.shape, test_label.shape)

pickle_file = 'gray.pickle'

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_data_c,
    'train_label': train_label,
    'test_dataset': test_data_c,
    'test_label': test_label,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise
