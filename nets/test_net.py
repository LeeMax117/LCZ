from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def get_net(inputs):

    # inputs is 32*32
    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.name_scope('conv1'):
        h_conv1 = tf.contrib.slim.conv2d(inputs, 32, [5, 5],
                                     padding='SAME',
                                     activation_fn=tf.nn.relu)

    # pooling into 16*16
    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = tf.contrib.slim.max_pool2d(h_conv1, [2, 2], stride=2,
                                         padding='VALID')

    # 16*16*64
    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope('conv2'):
        h_conv2 = tf.contrib.slim.conv2d(h_pool1, 64, [5, 5],
                                     padding='SAME',
                                     activation_fn=tf.nn.relu)

    # 8*8*64
    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = tf.contrib.slim.max_pool2d(h_conv2, [2, 2],
                                         stride=[2, 2], padding='VALID')

    # 1*1*1024
    with tf.name_scope('fc1'):
        h_pool2_flat = tf.contrib.slim.flatten(h_pool2)
        h_fc1 = tf.contrib.slim.fully_connected(h_pool2_flat, 1024, activation_fn=tf.nn.relu)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 17 classes, one for each digit
    # Map the 1024 features to 10 classes, one for each digit
    with tf.name_scope('fc2'):
        y = tf.squeeze(tf.contrib.slim.fully_connected(h_fc1, 17, activation_fn=None))

    return y,keep_prob