from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory
import pandas as pd
import os
import numpy as np
import linecache

def inference_once(file_path,placeholder,image,logit,endpoints):
    FLAGS = tf.app.flags.FLAGS

    saver = tf.train.Saver()
    sess = tf.Session()
    checkpoint_path = FLAGS.checkpoint_path
    saver.restore(sess, checkpoint_path)
    image_value = open(file_path, 'rb').read()
    logit_value = sess.run([logit], feed_dict={placeholder:image_value})

    print(logit_value)
    recall5_list=np.argsort(-logit_value[0][0])[:5]
    print(recall5_list)
    print(np.argmax(logit_value))

    return recall5_list

def gen_label_list(label_list_index,dataset_dir):

    label_list = []
    label_txt_addr = dataset_dir + '/labels.txt'

    for labels in label_list_index:
        label_txt = ''

        for label_index in labels:
            label_txt += (linecache.getline(label_txt_addr,label_index+1).split(':')[-1]).rstrip('\n')
            print(label_txt)

        label_list.append(label_txt)

    return label_list


def gen_csv(test_dir):
    label_list_index = []
    label_list = []
    filename_list  = os.listdir(test_dir)

    # define the scope of the inference produce
    tf.app.flags.DEFINE_string('dataset_name', 'quiz', '')
    tf.app.flags.DEFINE_string('dataset_dir', '/home/leemax/AI/xianxia/comp.jpg', '')

    tf.app.flags.DEFINE_string('model_name', 'inception_v4', '')
    tf.app.flags.DEFINE_string('output_file', './output.pb', '')

    tf.app.flags.DEFINE_string('checkpoint_path', '/home/leemax/AI/xianxia/train/model.ckpt-13732', '')
    # tf.app.flags.DEFINE_string('pic_path', file_path, '')

    FLAGS = tf.app.flags.FLAGS
    is_training = False
    preprocessing_name = FLAGS.model_name

    graph = tf.Graph().as_default()

    dataset = dataset_factory.get_dataset(FLAGS.dataset_name, 'train',
                                      FLAGS.dataset_dir)

    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training = False)

    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=dataset.num_classes,
        is_training=is_training)

    if hasattr(network_fn, 'default_image_size'):
        image_size = network_fn.default_image_size
    else:
        image_size = FLAGS.default_image_size
    placeholder = tf.placeholder(name='input', dtype=tf.string)
    image = tf.image.decode_jpeg(placeholder, channels=3)
    iamge = image_preprocessing_fn(image, image_size, image_size)
    image = tf.expand_dims(iamge, 0)
    logit, endpoints = network_fn(image)

    # gen the filename_list and label_list
    for filename in filename_list:
        file_path = test_dir + filename
        label_list_index.append(inference_once(file_path,placeholder,image,logit,endpoints))

    # generate label_list

    dataset_dir = FLAGS.dataset_dir
    label_list = gen_label_list(label_list_index,dataset_dir)

    # gen_csv
    dataframe = pd.DataFrame({'filename':filename_list,'label':label_list})
    dataframe.to_csv("test.csv",index=False,sep=',')

if __name__ == '__main__':
    test_dir = '/home/leemax/AI/xianxia/test/'
    gen_csv(test_dir)