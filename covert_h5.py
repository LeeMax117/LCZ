import os
import tensorflow as tf
import h5py
import matplotlib.pyplot as plt
import numpy as np

def bytes_feature(values):
  """Returns a TF-Feature of bytes.

  Args:
    values: A string.

  Returns:
    A TF-Feature.
  """
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def convert2TFrecord(data,outputfile):
    with tf.Graph().as_default():

        with tf.Session('') as sess:
            for s in ['sen1','sen2']:
                raw_data = data[s]
                for index in range(0,raw_data.shape[0]):

                    with tf.python_io.TFRecordWriter(outputfile) as tfrecord_writer:
                            # Read the filename:
                            image_data = raw_data[index,:,:,:]
                            height, width = image_data.shape[0],image_data.shape[1]

                            class_id = data['label'][index]

                            example = tf.train.Example(features=tf.train.Features(feature={
                                      'image/encoded': bytes_feature(image_data),
                                      'image/class/label': int64_feature(class_id),
                                      'image/height': int64_feature(height),
                                      'image/width': int64_feature(width),
                                  }))
                            tfrecord_writer.write(example.SerializeToString())


dataset_path = "D:\Documents\script\python_script\AI\competation"

### to change according to your machine
base_dir = os.path.expanduser("D:\\Documents\\script\\python_script\\AI\\competation")
# path_training = os.path.join(base_dir, 'training.h5')
path_validation = os.path.join(base_dir, 'validation.h5')

# print(path_training)
#fid_training = h5py.File(path_training,'r')
fid_validation = h5py.File(path_validation,'r')

output_filename = "./valadition.tfrecord"

print('converting to tfrecord')
convert2TFrecord(fid_validation,output_filename)
print('convert tfrecord finished')

