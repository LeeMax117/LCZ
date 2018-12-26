import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
import numpy


# for return_type:
# 0 - return mixed sen1 and sen2 as 18 channels
# 1 - return sen1 as 10 channels
# 2 - return sen2 as 8 channels
class DataSet:
    def __init__(self,
                 full_data,
                 resize = False,
                 resize_shape = None,
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

        assert (full_data['sen1'].shape[0] == full_data['label'].shape[0] \
                and full_data['sen2'].shape[0] == full_data['label'].shape[0]), ( \
                    'sen1.shape: %s  sen2.shape: %s labels.shape: %s' % \
                    (full_data['sen1'].shape, full_data['sen2'].shape, full_data['labels']))

        self._num_examples = full_data['sen1'].shape[0]

        self._labels = full_data['label']
        self._data_s1 = full_data['sen1']
        self._data_s2 = full_data['sen2']

        self._epochs_completed = 0
        self._index_in_epoch = 0

        self._return_type = return_type

        self._resize = resize
        if resize:
            self._shape = resize_shape
        else:
            self._shape = []
            self._shape.append(self._data_s1.shape[1])
            self._shape.append(self._data_s1.shape[2])

    @property
    def return_type(self):
        return self._return_type

    @property
    def resize(self):
        return self._resize

    @property
    def shape(self):
        return self._shape

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

    # for return_type is 1 or 2, return the specific 10 or 8 channels in sen1 or sen2
    def get_batch_data(self,data,start,end):

        # distinguish label and images
        if len(data.shape) < 4:
            batch_data = numpy.empty((end-start,data.shape[1]))
        else:
            # batch_data = numpy.zeros((end-start,data.shape[1],data.shape[2],data.shape[3]))
            batch_data = numpy.empty((end - start, self._shape[0], self._shape[1], data.shape[3]))
        for i , perm_index in enumerate(self._perm[start:end]):

            if self._resize:
                # batch_data[i] = tf.image.resize_bilinear(data[perm_index],self._shape)
                for l in range(0,7):
                    batch_data[i][l:3*l-1] = tf.image.resize_bilinear(data[perm_index][l:3*l-1],self._shape)
            else:
                batch_data[i] = data[perm_index]

        return batch_data

    # for return_type is 0, should concate 18 channels
    def get_conct_batch_data(self,start,end):

        shape_1 = self.data_s1.shape[1]
        shape_2 = self.data_s1.shape[2]
        shape_3 = self.data_s1.shape[3] + self.data_s2.shape[3]
        batch_data = numpy.empty((end-start,shape_1,shape_2,shape_3))
        labels = numpy.empty((end-start,self._labels.shape[1]))

        for i , perm_index in enumerate(self._perm[start:end]):
            batch_data[i] = numpy.concatenate((self.data_s1[perm_index],self.data_s2[perm_index]),axis = 2)
            labels[i] = self._labels[perm_index]

        if self._resize:
            batch_data = tf.image.resize_images(batch_data,self._shape)

        return batch_data,labels

    def next_batch(self,
                   batch_size,
                   shuffle=True):
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:

            # shuffle the dataset
            if shuffle:
                perm0 = numpy.arange(self._num_examples)
                numpy.random.shuffle(perm0)
                self._perm = perm0

        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start

            if self._return_type == 0:
                data_rest_part,labels_rest_part = self.get_conct_batch_data(start,self._num_examples)
            else:
                if self._return_type == 1:
                    data_s1_rest_part = self.get_batch_data(self._data_s1,start,self._num_examples)
                elif self._return_type == 2:
                    data_s2_rest_part = self.get_batch_data(self._data_s2,start,self._num_examples)
                labels_rest_part = self.get_batch_data(self._labels,start,self._num_examples)

            # Shuffle the data
            if shuffle:
                perm = numpy.arange(self._num_examples)
                numpy.random.shuffle(perm)
                self._perm = perm

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch

            if self._return_type == 0:
                data_new_part , labels_new_part = self.get_conct_batch_data(start,end)
                return numpy.concatenate((data_rest_part, data_new_part), axis=0), \
                           numpy.concatenate((labels_rest_part, labels_new_part), axis=0)
            else:
                labels_new_part = self.get_batch_data(self._labels, start, end)
                if self._return_type == 1:
                    data_s1_new_part = self.get_batch_data(self._data_s1,start,end)
                    return numpy.concatenate((data_s1_rest_part, data_s1_new_part), axis=0), \
                           numpy.concatenate((labels_rest_part, labels_new_part), axis=0)
                elif self._return_type == 2:
                    data_s2_new_part = self.get_batch_data(self._data_s2,start,end)
                    return numpy.concatenate((data_s2_rest_part, data_s2_new_part), axis=0), \
                           numpy.concatenate((labels_rest_part, labels_new_part), axis=0)

        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch

            # print('return_type:', self._return_type)
            if self._return_type == 0:
                return self.get_conct_batch_data(start,end)
            elif self._return_type == 1:
                return self.get_batch_data(self._data_s1,start,end), self.get_batch_data(self._labels, start, end)
            elif self._return_type == 2:
                return self.get_batch_data(self._data_s2,start,end), self.get_batch_data(self._labels, start, end)
