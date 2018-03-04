__author__ = 'ahmaddorri'

import scipy.io
import numpy as np
import pickle
import tensorflow as tf

import gc
gc.enable()

extra_data = scipy.io.loadmat('/Users/ahmaddorri/Downloads/extra_32x32.mat', variable_names='X').get('X')
extra_labels = scipy.io.loadmat('/Users/ahmaddorri/Downloads/extra_32x32.mat', variable_names='y').get('y')

extra_dataset = extra_data.transpose((3,0,1,2))
extra_label = extra_labels[:,0]

def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,:,:]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels
#valid_dataset, valid_label = randomize(valid_dataset, valid_label)