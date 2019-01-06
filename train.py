from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import h5py
import os
import tensorflow as tf
import json

from raw_data import DataSet
from nets import nets_factory
from nets.test_net import get_net

slim = tf.contrib.slim

def initialize_train_process(folder, json_path, data, is_training = True):
    if is_training:
        print('trainning from beginning')
    else:
        print('initialize validation...')

    if not os.path.exists(folder):
        os.mkdir(folder)
        print('create new ckpt folder in %s' % folder)

    with open(json_path, 'w') as f:
        dict_to_dump = {}
        # initialize the parameter to normalize
        data.normalize_data(batch_size=50000)
        dict_to_dump['average'] = data.average
        dict_to_dump['standard'] = data.standard
        json.dump(dict_to_dump, f)
    print('normalized para has been stored in %s' % json_path)

###########################################
##### set the parameter of the system######
###########################################

### to change according to your machine
base_dir = os.path.expanduser('D:\Documents\script\python_script\AI\competation\dataset')
# path_training = os.path.join(base_dir, 'training.h5')
path_validation = os.path.join(base_dir, 'validation.h5')
### set the checkpoint path
ckpt_folder = './model'
### set the train status data file path
json_path_train = os.path.join(ckpt_folder, 'train_data.json')
### set the validation status data file path
json_path_valid = os.path.join(ckpt_folder, 'valid_data.json')

# finetune_ckpt = 'D:\Documents\script\python_script\AI\competation\inception_v4.ckpt'
finetune_ckpt = None
# define the process of trainning or validation
is_training = True
train_from_begining = False
######################
###### end of define the parameters
#####################################

# print(path_training)
# fid_train = h5py.File(path_training, 'r')
fid_validation = h5py.File(path_validation, 'r')
# raw_data = DataSet(fid_validation,resize=True,resize_shape=[299,299])

raw_data = DataSet(fid_validation)

val_data = DataSet(fid_validation)

# Define loss and optimizer
x = tf.placeholder(tf.float32, [None, 32,32,18])
y_ = tf.placeholder(tf.float32, [None, 17])
#############################################################
# count the step
global_step = tf.Variable(0, trainable=False)
increment_op = tf.assign_add(global_step, tf.constant(1))

# The raw formulation of cross-entropy,
#
#   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
#                                 reduction_indices=[1]))
#
# can be numerically unstable.
#

network_fn = nets_factory.get_network_fn(
    'densenet',
    num_classes=17,
    weight_decay = 0.00004,
    is_training = is_training)


y, end_points = network_fn(x)

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
# l2_loss = tf.add_n( [tf.nn.l2_loss(w) for w in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)] )
# total_loss = cross_entropy + 7e-5*l2_loss

#########################################################################################
# exponential decay learning rate
# learning_rate = tf.train.exponential_decay(0.01, global_step, decay_steps=1, decay_rate=0.9997, staircase=False)
learning_rate = tf.placeholder(tf.float32)
train_step =  tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

## add loss and accuracy into summaries
# loss_summary = tf.summary.scalar('total_loss', total_loss)
lr_summary = tf.summary.scalar('learning_rate', learning_rate)
# l2_loss_summary = tf.summary.scalar('l2_loss', l2_loss)
cross_entropy_summary = tf.summary.scalar('cross_entropy', cross_entropy)
#####################################################################
## set training accuracy summary
train_acc_summary = tf.summary.scalar('train_acc', accuracy)

# limit the GPU usage
# tf_config = tf.ConfigProto()
# tf_config.gpu_options.per_process_gpu_memory_fraction = 0.75
# sess = tf.Session(config=tf_config)
sess = tf.Session()

init_op = tf.global_variables_initializer()
sess.run(init_op)

## setup summary workspace
summary_writer = tf.summary.FileWriter('logs', sess.graph)

## save the whole NN model into checkpoint path
saver = tf.train.Saver(max_to_keep=3)

# get fintune from the downloaded pre-train ckpt
if finetune_ckpt and is_training:
    exclude = ['InceptionV4/Logits','Conv2d_1a_3x3','Conv2d_2a_3x3','Conv2d_2b_3x3','Mixed_3a','InceptionV4/Mixed_4a','InceptionV4/AuxLogits']
    variables_to_restore = slim.get_variables_to_restore(exclude=exclude)
    # print(variables_to_restore)
    saver_inception = tf.train.Saver(variables_to_restore)
    saver_inception.restore(sess, finetune_ckpt)
    print('load weight from %s'%finetune_ckpt)

# for not get pre-train ckpt
else:
    model_file = tf.train.latest_checkpoint(ckpt_folder)

    if is_training:
        if train_from_begining:
            if not os.path.exists(json_path_train):
                initialize_train_process(ckpt_folder, json_path_train, raw_data)
            else:
                # load avg and standard from json
                with open(json_path_train, 'r') as file:
                    train_data_paras = json.load(file)
                    raw_data.set_normalize_para(train_data_paras['average'], train_data_paras['standard'])
        else:
            try:
                saver.restore(sess, model_file)
            except ValueError:
                if not os.path.exists(ckpt_folder):
                    initialize_train_process(ckpt_folder, json_path_train, raw_data)

            try:
                with open(json_path_train, 'r') as file:
                    train_data_paras = json.load(file)
                    raw_data.set_normalize_para(train_data_paras['average'], train_data_paras['standard'])
                    # glb_step = train_data_paras['glb_step']
                    checkpoint = os.path.join(ckpt_folder, 'checkpoint')
                    if os.path.exists(checkpoint):
                        raw_data.set_ind_in_epoch(train_data_paras['ind_in_epoch'])
                        raw_data.set_epochs_completed(train_data_paras['epochs_completed'])
                        raw_data.set_random_list(train_data_paras['random_list'])
                    else:
                        print('train from begining with loaded json')
                        print('index in epoch is %d, epochs completed is %d'%(raw_data.index_in_epoch,raw_data.epochs_completed))
            except (IOError) as e:
                initialize_train_process(ckpt_folder, json_path_train, raw_data)
    else:
        saver.restore(sess, model_file)
        ### load validation status data from json file
        try:
            with open(json_path_train, 'r') as file:
                valid_data_paras = json.load(file)
                val_data.set_normalize_para(valid_data_paras['average'], valid_data_paras['standard'])
        except (IOError) as e:
            initialize_train_process(ckpt_folder, json_path_valid, val_data, is_training = False)

#####################################################################
## get the initial summaries
summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

# Train
lr = 0.0001
if is_training:

    ## if is the beginning of the training then discard all old summaries.
    if train_from_begining:
        summaries.clear()

    #summaries.add(loss_summary)
    summaries.add(lr_summary)
    #summaries.add(l2_loss_summary)
    summaries.add(train_acc_summary)
    summaries.add(cross_entropy_summary)
    summary_op = tf.summary.merge(list(summaries), name='summary_op')
    #####################################################################

    last_decay_epoch = 0
    while raw_data.epochs_completed < 30:
        batch_xs, batch_ys = raw_data.next_batch(12)
        # if use resize, need to transer tensor to numpy
        if raw_data.resize:
            batch_xs = batch_xs.eval(session=sess)

        _, summary, loss, steps = sess.run(
            [train_step, summary_op, cross_entropy, increment_op],
            feed_dict={x: batch_xs, y_: batch_ys, learning_rate: lr})


        if steps % 1 == 0:
            print('step %d, entropy loss: %f' %
                  (steps, loss))

        if steps % 100 == 0:
            ###################1234###################################
            ## add summary 1 to file
            summary_writer.add_summary(summary, global_step=steps)

            # save ckpt
            ckpt_path = os.path.join(ckpt_folder, 'model.ckpt')
            saver.save(sess, ckpt_path , global_step=steps)
			# save raw data parameters.
            with open(json_path_train, 'w') as file:
                jdict = dict()
                jdict['average'] = raw_data.average
                jdict['standard'] = raw_data.standard
                jdict['ind_in_epoch'] = raw_data.index_in_epoch
                jdict['epochs_completed'] = raw_data.epochs_completed
                jdict['random_list'] = raw_data.random_list.tolist()
                json.dump(jdict, file)
            print('Model saved in path %s' % ckpt_path)
            print('learning_rate is %f'%lr)


        if raw_data.epochs_completed % 3 == 0 and raw_data.epochs_completed != last_decay_epoch:
            last_decay_epoch = raw_data.epochs_completed
            lr = lr * 0.618


    summary_writer.close()
else:
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print('start inference from %d data'%val_data.index_in_epoch)
    correct_num = 0
    while not val_data.epochs_completed:
        batch_xs, batch_ys = val_data.next_batch(300, is_training = is_training)
        acc = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys, keep_prob:0.5})
        correct_num += batch_xs.shape[0] * acc
        print(acc)

    print('the whole acc is %f'%(correct_num/val_data.num_examples))
