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

    def get_batch_data(self,data,start,end):

        # distinguish label and images
        if len(data.shape) < 4:
            batch_data = numpy.empty((end-start,data.shape[1]))
        else:
            batch_data = numpy.zeros((end-start,data.shape[1],data.shape[2],data.shape[3]))
        for i , perm_index in enumerate(self._perm[start:end]):
            batch_data[i] = data[perm_index]

        return batch_data


    def next_batch(self, batch_size, shuffle=True):
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
            data_s1_rest_part = self.get_batch_data(self._data_s1,start,self.num_examples)
            data_s2_rest_part = self.get_batch_data(self._data_s2,start,self.num_examples)
            labels_rest_part = self.get_batch_data(self._labels,start,self.num_examples)
            # Shuffle the data
            if shuffle:
                perm = numpy.arange(self._num_examples)
                numpy.random.shuffle(perm)
                self._perm = perm

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            data_s1_new_part = self.get_batch_data(self._data_s1,start,end)
            data_s2_new_part = self.get_batch_data(self._data_s2,start,end)
            labels_new_part = self.get_batch_data(self._labels,start,end)

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
                return self.get_batch_data(self._data_s1,start,end), self.get_batch_data(self._data_s2,start,end), \
                       self.get_batch_data(self._labels, start, end)
            elif self._return_type == 1:
                return self.get_batch_data(self._data_s1,start,end), self.get_batch_data(self._labels, start, end)
            elif self._return_type == 2:
                return self.get_batch_data(self._data_s2,start,end), self.get_batch_data(self._labels, start, end)
