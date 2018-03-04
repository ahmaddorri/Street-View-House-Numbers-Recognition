__author__ = 'ahmaddorri'

import tensorflow as tf
import pickle

pickle_file="GCN.pickle"

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_label = save['train_label']
    test_dataset = save['test_dataset']
    test_label = save['test_label']
    del save  # hint to help gc free up memory
    print('Training set', train_dataset.shape, train_label.shape)
    print('Test set', test_dataset.shape, test_label.shape)


print(tf.shape(train_dataset))
image_size = 32
num_labels = 11 # 0-9, + blank
num_channels = 1 # grayscale

import numpy as np

def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)

    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels


train_dataset, train_label = reformat(train_dataset, train_label)
test_dataset, test_label = reformat(test_dataset, test_label)
#valid_dataset, valid_label = reformat(valid_dataset, valid_label)


print('...Training set', train_dataset.shape, train_label.shape)
#print('...Test set', test_dataset.shape, test_label.shape)

#print(len(train_dataset[0]))
#print(train_dataset[0])

#print(len(train_dataset[0][0]))
#print(train_dataset[0][0])


#######################################################################################

batch_size=50

x_image = tf.placeholder(tf.float32, shape=[batch_size,image_size,image_size,num_channels])
y_ = tf.placeholder(tf.float32, [None, 11])

w_conv1=tf.Variable(tf.truncated_normal([5,5,1,16], stddev=0.1))
conv1 = tf.nn.conv2d(input=x_image,filter=w_conv1,strides=[1,1,1,1],padding="SAME")
b_conv1 = tf.Variable(tf.constant(0.1, shape=[16]))
h_conv1 = tf.nn.relu(conv1 + b_conv1)
h_pool1 = tf.nn.max_pool(value=h_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

w_conv2=tf.Variable(tf.truncated_normal([5,5,16,32], stddev=0.1))
conv2 = tf.nn.conv2d(input=h_pool1,filter=w_conv2,strides=[1,1,1,1],padding="SAME")
b_conv2 = tf.Variable(tf.constant(0.1, shape=[32]))
h_conv2 = tf.nn.relu(conv2 + b_conv2)
h_pool2 = tf.nn.max_pool(value=h_conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")


#Densely Connected Layer

W_fc1 = tf.Variable(tf.truncated_normal([8 * 8 * 32, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))


h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*32])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


#Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#Readout layer
W_fc2 = tf.Variable(tf.truncated_normal([1024, 11], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[11]))

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def getBacth(step,data,lable):
    offset = (step * batch_size) % (lable.shape[0] - batch_size)
    batch_data = data[offset:(offset + batch_size), :, :, :]
    batch_labels = lable[offset:(offset + batch_size), :]
    return batch_data,batch_labels


num_steps=10001

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(num_steps):
        offset = (step * batch_size) % (train_label.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_label[offset:(offset + batch_size), :]
        if step % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x_image: batch_data, y_: batch_labels, keep_prob: 1.0})
            print('step %d, training accuracy %g' % (step, train_accuracy))
        #dropOutProb = 0.8/np.log10(step+2)
        train_step.run(feed_dict={x_image: batch_data, y_: batch_labels, keep_prob: 0.5})

    n_batches = test_dataset.shape[0] // batch_size
    cumulative_accuracy = 0.0
    for index in range(n_batches):
        batch_data , batch_lables = getBacth(index,test_dataset,test_label)
        cumulative_accuracy += accuracy.eval(feed_dict={x_image: batch_data, y_: batch_lables, keep_prob: 1.0})
    print("test accuracy {}".format(cumulative_accuracy / n_batches))


