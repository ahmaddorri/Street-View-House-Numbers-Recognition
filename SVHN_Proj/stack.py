__author__ = 'ahmaddorri'

import scipy.io


train_data = scipy.io.loadmat('/Users/ahmaddorri/Desktop/NN_proj/train_32x32.mat', variable_names='X').get('X')
train_labels = scipy.io.loadmat('/Users/ahmaddorri/Desktop/NN_proj/train_32x32.mat', variable_names='y').get('y')
test_data = scipy.io.loadmat('/Users/ahmaddorri/Desktop/NN_proj/test_32x32.mat', variable_names='X').get('X')
test_labels = scipy.io.loadmat('/Users/ahmaddorri/Desktop/NN_proj/test_32x32.mat', variable_names='y').get('y')
#extra_data = scipy.io.loadmat('/Users/ahmaddorri/Downloads/extra_32x32.mat', variable_names='X').get('X')
#extra_labels = scipy.io.loadmat('/Users/ahmaddorri/Downloads/extra_32x32.mat', variable_names='y').get('y')

print(train_data.shape, train_labels.shape)
print(test_data.shape, test_labels.shape)

train_dataset = train_data.transpose((3,0,1,2))
train_label = train_labels[:,0]
test_dataset = test_data.transpose((3,0,1,2))
test_label = test_labels[:,0]


print(train_dataset.shape, train_label.shape)
print(test_dataset.shape, test_label.shape)