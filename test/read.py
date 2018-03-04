__author__ = 'ahmaddorri'

import matplotlib.image as img
import numpy as np


import scipy.io
train_data = scipy.io.loadmat('/Users/ahmaddorri/Desktop/NN_proj/train_32x32.mat', variable_names='X').get('X')

train_dataset = train_data.transpose((3,0,1,2))

import scipy.misc
scipy.misc.imsave('test/outfile2.jpg', train_dataset[290])

a = "test/outfile2.jpg"
image = img.imread(a)
data = []
data.append(image)
data=np.array(data)

print(data.shape)
