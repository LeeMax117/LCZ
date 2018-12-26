from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import h5py
import os
import tensorflow as tf

from raw_data import DataSet
from net.test_net import get_net

### to change according to your machine
base_dir = os.path.expanduser('D:\Documents\script\python_script\AI\competation\dataset')
# path_training = os.path.join(base_dir, 'training.h5')
path_validation = os.path.join(base_dir, 'validation.h5')

# print(path_training)
#fid_training = h5py.File(path_training,'r')
fid_validation = h5py.File(path_validation,'r')

# Define loss and optimizer
x = tf.placeholder(tf.float32, [None, 32,32,18])
y_ = tf.placeholder(tf.float32, [None, 17])
learning_rate = tf.placeholder(tf.float32)

raw_data = DataSet(fid_validation)

# The raw formulation of cross-entropy,
#
#   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
#                                 reduction_indices=[1]))
#
# can be numerically unstable.
#
# So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
# outputs of 'y', and then average across the batch.
y,keep_prob = get_net(x)

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])
total_loss = cross_entropy + 7e-5 * l2_loss
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

## add loss and accuracy into summaries
loss_summary = tf.summary.scalar('total_loss', total_loss)
acc_summary = tf.summary.scalar('accuracy', accuracy)
## setup summary operation
sum_ops_1 = tf.summary.merge([loss_summary])
sum_ops_2 = tf.summary.merge([acc_summary])

sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)

## setup summary workspace
summary_writer = tf.summary.FileWriter('logs', sess.graph)

# Train
# Initialized learning rate
lr = 0.618
for step in range(3000):
    batch_xs, batch_ys = raw_data.next_batch(100)
    _, summary_1, loss, l2_loss_value, total_loss_value = sess.run(
        [train_step, sum_ops_1, cross_entropy, l2_loss, total_loss],
        feed_dict={x: batch_xs, y_: batch_ys, learning_rate: lr, keep_prob : 0.5})
    ## add summary 1 to file
    summary_writer.add_summary(summary_1, global_step=step)

    if (step + 1) % 100 == 0:
        print('y:',y)
        print('step %d, entropy loss: %f, l2_loss: %f, total loss: %f' %
              (step + 1, loss, l2_loss_value, total_loss_value))
        # Test trained model
        _, summary_2 = sess.run([accuracy, sum_ops_2], feed_dict={x: batch_xs, y_: batch_ys, keep_prob:0.5})
        ## add summary 2 to file
        summary_writer.add_summary(summary_2, global_step=step)

    # learning rate dacey
    if (step + 1) % 300 == 0:
        lr = 0.618 * lr
        
summary_writer.close()