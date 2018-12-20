from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
import numpy

import tensorflow as tf

### to change according to your machine
base_dir = os.path.expanduser('./dataset')
# path_training = os.path.join(base_dir, 'training.h5')
path_validation = os.path.join(base_dir, 'validation.h5')

# print(path_training)
#fid_training = h5py.File(path_training,'r')
fid_validation = h5py.File(path_validation,'r')


# for return_type:
# 0 - return mixed sen1 and sen2 as 18 channels
# 1 - return sen1 as 10 channels
# 2 - return sen2 as 8 channels
class DataSet:
    def __init__(self,
                 full_data,
                 return_type=0,
                 dtype=dtypes.float32,
                 seed=None):

        seed1, seed2 = random_seed.get_seed(seed)
        # If op level seed is not set, use whatever graph level seed is returned
        numpy.random.seed(seed1 if seed is None else seed2)
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                            dtype)

        # transform the hdf5 format into numpy
        data_s1 = numpy.array(full_data['sen1'])
        # print('load s1 complete')
        data_s2 = numpy.array(full_data['sen2'])
        # print('load s2 complete')
        labels = numpy.array(full_data['label'])
        # print('load label complete')
        # print('already transform into numpy array')

        assert (data_s1.shape[0] == labels.shape[0] \
                and data_s2.shape[0] == labels.shape[0]), ( \
                    'sen1.shape: %s  sen2.shape: %s labels.shape: %s' % \
                    (data_s1.shape, data_s2.shape, labels.shape))

        self._num_examples = full_data['sen1'].shape[0]

        self._labels = labels
        self._data_s1 = data_s1
        self._data_s2 = data_s2

        self._epochs_completed = 0
        self._index_in_epoch = 0

        self._return_type = return_type

    @property
    def return_type(self):
        return self._return_type

    @property
    def data_s1(self):
        return self._data_s1

    @property
    def data_s2(self):
        return self._data_s2

    @property
    def label(self):
        return self._label

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = numpy.arange(self._num_examples)
            # perm0 = numpy.arange(1000)
            numpy.random.shuffle(perm0)
            # print('perm0',perm0)
            self._data_s1 = self.data_s1[perm0]
            self._data_s2 = self.data_s2[perm0]
            self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            data_s1_rest_part = self._data_s1[start:self._num_examples]
            data_s2_rest_part = self._data_s2[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = numpy.arange(self._num_examples)
                numpy.random.shuffle(perm)
                self._data_s1 = self.data_s1[perm]
                self._data_s2 = self.data_s2[perm]
                self._labels = self.labels[perm]

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            data_s1_new_part = self._data_s1[start:end]
            data_s2_new_part = self._data_s2[start:end]
            labels_new_part = self._labels[start:end]

            if self._return_type == 0:
                return numpy.concatenate((data_s1_rest_part, data_s1_new_part), axis=0), \
                       numpy.concatenate((data_s2_rest_part, data_s2_new_part), axis=0), \
                       numpy.concatenate((labels_rest_part, labels_new_part), axis=0)
            elif self._return_type == 1:
                return numpy.concatenate((data_s1_rest_part, data_s1_new_part), axis=0), \
                       numpy.concatenate((labels_rest_part, labels_new_part), axis=0)
            elif self._return_type == 2:
                return numpy.concatenate((data_s2_rest_part, data_s2_new_part), axis=0), \
                       numpy.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch

            # print('return_type:', self._return_type)
            if self._return_type == 0:
                return self._data_s1[start:end], self._data_s2[start:end], self._labels[start:end]
            elif self._return_type == 1:
                return self._data_s1[start:end], self._labels[start:end]
            elif self._return_type == 2:
                return self._data_s2[start:end], self._labels[start:end]


# Define loss and optimizer
x = tf.placeholder(tf.float32, [None, 32,32,8])
y_ = tf.placeholder(tf.float32, [None, 17])
learning_rate = tf.placeholder(tf.float32)

# reshape the n*784 dense input into n*28*28*1
with tf.name_scope('reshape'):
    x_image = x

# First convolutional layer - maps one grayscale image to 32 feature maps.
with tf.name_scope('conv1'):
    h_conv1 = tf.contrib.slim.conv2d(x_image, 32, [5, 5],
                                     padding='SAME',
                                     activation_fn=tf.nn.relu)

# Pooling layer - downsamples by 2X.
with tf.name_scope('pool1'):
    h_pool1 = tf.contrib.slim.max_pool2d(h_conv1, [2, 2], stride=2,
                                         padding='VALID')

# Second convolutional layer -- maps 32 feature maps to 64.
with tf.name_scope('conv2'):
    h_conv2 = tf.contrib.slim.conv2d(h_pool1, 64, [5, 5],
                                     padding='SAME',
                                     activation_fn=tf.nn.relu)

# Second pooling layer.
with tf.name_scope('pool2'):
    h_pool2 = tf.contrib.slim.max_pool2d(h_conv2, [2, 2],
                                         stride=[2, 2], padding='VALID')

# Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
# is down to 7x7x64 feature maps -- maps this to 1024 features.
with tf.name_scope('fc1'):
    h_pool2_flat = tf.contrib.slim.avg_pool2d(h_pool2, h_pool2.shape[1:3],
                                              stride=[1, 1], padding='VALID')
    h_fc1 = tf.contrib.slim.conv2d(h_pool2_flat, 1024, [1, 1], activation_fn=tf.nn.relu)

# Dropout - controls the complexity of the model, prevents co-adaptation of
# features.
with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Map the 1024 features to 10 classes, one for each digit
with tf.name_scope('fc2'):
    y = tf.squeeze(tf.contrib.slim.conv2d(h_fc1_drop, 17, [1, 1], activation_fn=None))


raw_data = DataSet(fid_validation,1)

# The raw formulation of cross-entropy,
#
#   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
#                                 reduction_indices=[1]))
#
# can be numerically unstable.
#
# So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
# outputs of 'y', and then average across the batch.
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])
total_loss = cross_entropy + 7e-5 * l2_loss
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)

sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)
# Train
# Initialized learning rate
lr = 0.618
for step in range(3000):
    batch_xs, batch_ys = raw_data.next_batch(100)
    _, loss, l2_loss_value, total_loss_value = sess.run(
        [train_step, cross_entropy, l2_loss, total_loss],
        feed_dict={x: batch_xs, y_: batch_ys, learning_rate: lr, keep_prob: 0.5})

    if (step + 1) % 100 == 0:
        print('step %d, entropy loss: %f, l2_loss: %f, total loss: %f' %
              (step + 1, loss, l2_loss_value, total_loss_value))
        # Test trained model
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5}))

    # learning rate dacey
    if (step + 1) % 300 == 0:
        lr = 0.618 * lr

    if (step + 1) % 1000 == 0:
        print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                            y_: mnist.test.labels, keep_prob: 0.5}))
