from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import h5py
import os
import tensorflow as tf

from raw_data import DataSet
from nets import nets_factory

slim = tf.contrib.slim

###########################################
##### set the parameter of the system######
###########################################

### to change according to your machine
base_dir = os.path.expanduser('D:\Documents\script\python_script\AI\competation\dataset')
# path_training = os.path.join(base_dir, 'training.h5')
path_validation = os.path.join(base_dir, 'validation.h5')
### set the checkpoint path
ckpt_folder = 'model'
finetune_ckpt = None
# define the process of trainning or validation
is_trainning = True

# print(path_training)
#fid_training = h5py.File(path_training,'r')
fid_validation = h5py.File(path_validation,'r')

# Define loss and optimizer
x = tf.placeholder(tf.float32, [None, 32,32,18])
y_ = tf.placeholder(tf.float32, [None, 17])
learning_rate = tf.placeholder(tf.float32)
global_step = tf.Variable(0, name='step')

# raw_data = DataSet(fid_validation,resize=True,resize_shape=[299,299])
raw_data = DataSet(fid_validation)

# The raw formulation of cross-entropy,
#
#   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
#                                 reduction_indices=[1]))
#
# can be numerically unstable.
#
network_fn = nets_factory.get_network_fn(
    'M_inception_v4',
    num_classes=17,
    weight_decay=0.0001,
    is_training=is_trainning)

y , end_points = network_fn(x)

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])
# total_loss = cross_entropy + 7e-5 * l2_loss
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

## add loss and accuracy into summaries
loss_summary = tf.summary.scalar('total_loss', cross_entropy)
acc_summary = tf.summary.scalar('accuracy', accuracy)
## setup summary operation
sum_ops_1 = tf.summary.merge([loss_summary])
sum_ops_2 = tf.summary.merge([acc_summary])

sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)

## setup summary workspace
summary_writer = tf.summary.FileWriter('logs', sess.graph)

## save the whole NN model into checkpoint path
saver = tf.train.Saver(max_to_keep=3)

if finetune_ckpt:
    saver.restore(sess, finetune_ckpt)
else:
    model_file = tf.train.latest_checkpoint(ckpt_folder)
    try:
        saver.restore(sess, model_file)
    except ValueError:
        if os.path.isdir(ckpt_folder):
            print('trainning from beginning')

# Train
if is_trainning:
    # Initialized learning rate
    lr = 0.01
    for step in range(300):
        batch_xs, batch_ys = raw_data.next_batch(12)

        global_step = global_step + 1
        # if use resize, need to transer tensor to numpy
        if raw_data.resize:
            batch_xs = batch_xs.eval(session=sess)

        _, summary_1, loss, g_step = sess.run(
            [train_step, sum_ops_1, cross_entropy, global_step],
            feed_dict={x: batch_xs, y_: batch_ys, learning_rate: lr})

        print('step %d, entropy loss: %f' %
              (g_step, loss))

        if (step + 1) % 2 == 0:
            ## add summary 1 to file
            summary_writer.add_summary(summary_1, global_step=step)

            # save ckpt
            ckpt_path = os.path.join(ckpt_folder, 'model.ckpt')
            saver.save(sess, ckpt_path , global_step=step)
            print('Model saved in path %s' % ckpt_path)
            # Test trained model
            # acc, summary_2 = sess.run([accuracy, sum_ops_2], feed_dict={x: batch_xs, y_: batch_ys})
            # print(acc)
            ## add summary 2 to file
            #summary_writer.add_summary(summary_2, global_step=step)
            summary_writer.close()

        # learning rate dacey
        if (step + 1) % 30 == 0:
            lr = 0.618 * lr
else:
    correct_prediction = tf.equal(tf.argmax(logit, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    batch_xs, batch_ys = raw_data.next_batch(24)
    acc = sess.run(accuracy, feed_dict={placeholder: batch_xs, y_: batch_ys})
    print(acc)
