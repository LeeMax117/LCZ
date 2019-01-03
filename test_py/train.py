from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import h5py
import os
import tensorflow as tf
import json

from raw_data import DataSet
from nets import nets_factory

slim = tf.contrib.slim

###########################################
##### set the parameter of the system######
###########################################

### to change according to your machine
base_dir = os.path.expanduser('D:/test')
# path_training = os.path.join(base_dir, 'training.h5')
path_validation = os.path.join(base_dir, 'validation.h5')
### set the checkpoint path
ckpt_folder = 'model'
### set the train status data file path
json_path = os.path.join(ckpt_folder, 'train_data.json')

# finetune_ckpt = 'D:\Documents\script\python_script\AI\competation\inception_v4.ckpt'
finetune_ckpt = None
# define the process of trainning or validation
is_trainning = True
######################
###### end of define the parameters
#####################################

# print(path_training)
#fid_training = h5py.File(path_training,'r')
fid_validation = h5py.File(path_validation,'r')

# Define loss and optimizer
x = tf.placeholder(tf.float32, [None, 32,32,18])
y_ = tf.placeholder(tf.float32, [None, 17])
#############################################################
# 迭代计数器
global_step = tf.Variable(0, trainable=False)               #
# 迭代+1
increment_op = tf.assign_add(global_step, tf.constant(1))#
############################################################
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
    is_training=is_trainning)

y , end_points = network_fn(x)

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])
# total_loss = cross_entropy + 7e-5 * l2_loss
#########################################################################################
# exponential decay learning rate
learning_rate = tf.train.exponential_decay(0.1, global_step, decay_steps=1, decay_rate=0.9, staircase=False)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
#########################################################################################
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

############################################################################
## add loss and accuracy into summaries
loss_summary = tf.summary.scalar('total_loss', cross_entropy)
lr_summary = tf.summary.scalar('learning_rate', learning_rate)
acc_summary = tf.summary.scalar('accuracy', accuracy)
############################################################################
## setup summary operation
#sum_ops_1 = tf.summary.merge([loss_summary])
#sum_ops_2 = tf.summary.merge([acc_summary])

sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)

## setup summary workspace
summary_writer = tf.summary.FileWriter('logs', sess.graph)

## save the whole NN model into checkpoint path
saver = tf.train.Saver(max_to_keep=3)

glb_step = 0

if finetune_ckpt and is_trainning:
    exclude = ['InceptionV4/Logits','Conv2d_1a_3x3','Conv2d_2a_3x3','Conv2d_2b_3x3','Mixed_3a','InceptionV4/Mixed_4a','InceptionV4/AuxLogits']
    variables_to_restore = slim.get_variables_to_restore(exclude=exclude)
    # print(variables_to_restore)
    saver_inception = tf.train.Saver(variables_to_restore)
    saver_inception.restore(sess, finetune_ckpt)
    print('load weight from %s'%finetune_ckpt)
else:
    model_file = tf.train.latest_checkpoint(ckpt_folder)
    try:
        saver.restore(sess, model_file)
    except ValueError:
        if os.path.isdir(ckpt_folder) or not os.path.exists(ckpt_folder):
            print('trainning from beginning')
            if not os.path.exists(ckpt_folder):
                print('create new ckpt folder in %s'%ckpt_folder)
	
	### load training status data from json file
    try:
        with open(json_path, 'r') as file:
            train_data_paras = json.load(file)
            random_list = train_data_paras.get('random_list', None)
            if random_list is None:
                glb_step = 0
                raw_data.set_ind_in_epoch(0)
                raw_data.set_epochs_completed(0)
            else:
                glb_step = train_data_paras.get('glb_step', 0)
                raw_data.set_ind_in_epoch(train_data_paras.get('ind_in_epoch', 0))
                raw_data.set_epochs_completed(train_data_paras.get('epochs_completed', 0))
                raw_data.set_random_list(random_list)
                
    except (IOError) as e:
        print('create new json file')

#####################################################################
## get the initial summaries
summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
summaries.add(loss_summary)
summaries.add(lr_summary)
summary_op = tf.summary.merge(list(summaries), name='summary_op')
#####################################################################

# Train
if is_trainning:
    # Initialized learning rate
    step = glb_step
    while True:
        step += 1
        batch_xs, batch_ys = raw_data.next_batch(12)
        # if use resize, need to transer tensor to numpy
        if raw_data.resize:
            batch_xs = batch_xs.eval(session=sess)

        ###################################################################
        _, summary, loss, _ = sess.run(
            [train_step, summary_op, cross_entropy, increment_op],
            feed_dict={x: batch_xs, y_: batch_ys})
        ###################################################################

        print('step %d, entropy loss: %f' %
              (step, loss))

        if step % 2 == 0:
            ######################################################
            ## add summary 1 to file
            summary_writer.add_summary(summary, global_step=step)
            ######################################################

            # save ckpt
            ckpt_path = os.path.join(ckpt_folder, 'model.ckpt')
            saver.save(sess, ckpt_path , global_step=step)
			# save raw data parameters.
            with open(json_path, 'w') as file:
                jdict = dict()
                jdict['glb_step'] = step
                jdict['ind_in_epoch'] = raw_data.index_in_epoch
                jdict['epochs_completed'] = raw_data.epochs_completed
                jdict['random_list'] = raw_data.random_list.tolist()
                json.dump(jdict, file)
            print('Model saved in path %s' % ckpt_path)
            # Test trained model
            # acc, summary_2 = sess.run([accuracy, sum_ops_2], feed_dict={x: batch_xs, y_: batch_ys})
            # print(acc)
            ## add summary 2 to file
            #summary_writer.add_summary(summary_2, global_step=step)

else:
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print('start inference from %d data'%raw_data.index_in_epoch)
    while not raw_data.epochs_completed:
        batch_xs, batch_ys = raw_data.next_batch(24,is_trainning = is_trainning)
        acc = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
        print(acc)

## close the summary writer
summary_writer.close()