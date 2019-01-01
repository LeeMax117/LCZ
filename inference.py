from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import pandas as pd
import h5py
import os
from raw_data import DataSet
from nets import nets_factory

### to change according to your machine
base_dir = os.path.expanduser('D:\Documents\script\python_script\AI\competation\dataset')
# path_training = os.path.join(base_dir, 'training.h5')
path_validation = os.path.join(base_dir, 'validation.h5')
#fid_training = h5py.File(path_training,'r')
fid_validation = h5py.File(path_validation,'r')

raw_data = DataSet(fid_validation)

def inference_once(placeholder,logit,batch_size):

    saver = tf.train.Saver()
    sess = tf.Session()
    checkpoint_path = 'model/model.ckpt-30'
    saver.restore(sess, checkpoint_path)

    y_ = tf.placeholder(tf.float32, [None, 17])

    batch_xs, batch_ys = raw_data.next_batch(batch_size)

    logit_value = sess.run([logit], feed_dict={placeholder:batch_xs})

    correct_prediction = tf.equal(tf.argmax(logit, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    acc = sess.run(accuracy, feed_dict={placeholder:batch_xs, y_: batch_ys})
    print(acc)

    return logit_value


is_training = False

placeholder = tf.placeholder(tf.float32, [None, 32,32,18])

network_fn = nets_factory.get_network_fn(
    'M_inception_v4',
    num_classes=17,
    is_training=is_training)

logit , end_points = network_fn(placeholder)

batch_size = 24
logits = inference_once(placeholder,logit,batch_size)

list_predict = []
for i in logits[0]:
    onehot = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    onehot[np.argmax(i)] = 1
    list_predict.append(onehot)
print(list_predict)

dataframe = pd.DataFrame(list_predict)
print(dataframe)
dataframe.to_csv("test.csv",index=False,sep=',',header=None)