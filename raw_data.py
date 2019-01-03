import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
import numpy
from nomarlize_dataset import get_avg_standard

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
                 inference = False,
                 seed=None):

        seed1, seed2 = random_seed.get_seed(seed)
        # If op level seed is not set, use whatever graph level seed is returned
        numpy.random.seed(seed1 if seed is None else seed2)
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                            dtype)

        if inference:
            assert(full_data['sen1'].shape[0] == full_data['sen2'].shape[0])
        else:
            assert (full_data['sen1'].shape[0] == full_data['label'].shape[0] \
                    and full_data['sen2'].shape[0] == full_data['label'].shape[0]), ( \
                        'sen1.shape: %s  sen2.shape: %s labels.shape: %s' % \
                        (full_data['sen1'].shape, full_data['sen2'].shape, full_data['labels']))

        self._num_examples = full_data['sen1'].shape[0]

        if not inference:
            self._labels = full_data['label']
        else:
            self._labels = None

        self._data_s1 = full_data['sen1']
        self._data_s2 = full_data['sen2']

        self._inference = inference

        self._epochs_completed = 0
        self._index_in_epoch = 0

        self._return_type = return_type

        # use resize to resize the image
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
    def inference(self):
        return self._inference

    @property
    def index_in_epoch(self):
        return self._index_in_epoch

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
    def labels(self):
        if self._inference:
            return None
        else:
            return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed
    
    @property
    def random_list(self):
        return self._perm

    @property
    def average(self):
        return self._average

    @property
    def standard(self):
        return self._standard
    
    def set_ind_in_epoch(self, ind_in_epoch):
        self._index_in_epoch = ind_in_epoch
        
    def set_epochs_completed(self, epochs_completed):
        self._epochs_completed = epochs_completed
        
    def set_random_list(self, random_list):
        self._perm = numpy.array(random_list)

    def set_normalize_para(self,average,standard):
        self._average = average
        self._standard = standard

    def normalize_data(self):
        print('starting normalize the dataset')
        if self.return_type == 0:
            # get s1 avg and standard first
            average, standard = get_avg_standard(self._data_s1)
            s2_average, s2_standard = get_avg_standard(self._data_s2)
            average.extend(s2_average)
            standard.extend(s2_standard)

            self._average = average
            self._standard = standard

            print('normalized done')

        else:
            print('not surpport this return_type yet.....')


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
        if self._labels:
            labels = numpy.empty((end-start,self._labels.shape[1]))
        else:
            labels = None

        for i , perm_index in enumerate(self._perm[start:end]):
            batch_data[i] = numpy.concatenate((self.data_s1[perm_index],self.data_s2[perm_index]),axis = 2)
            if self._labels:
                labels[i] = self._labels[perm_index]

        if self._resize:
            batch_data = tf.image.resize_images(batch_data,self._shape)

        # normalize the data
        batch_data = (batch_data - self._average)/self._standard

        return batch_data,labels

    def next_batch(self,
                   batch_size,
                   shuffle=True,
                   is_trainning = True):
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
                if self._inference or (not is_trainning):
                    print(data_rest_part.shape)
                    return data_rest_part
            else:
                if self._return_type == 1:
                    data_s1_rest_part = self.get_batch_data(self._data_s1,start,self._num_examples)
                elif self._return_type == 2:
                    data_s2_rest_part = self.get_batch_data(self._data_s2,start,self._num_examples)
                labels_rest_part = self.get_batch_data(self._labels,start,self._num_examples)
                if not is_trainning:
                    return data_rest_part, labels_rest_part

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
                batch_data , batch_labels = self.get_conct_batch_data(start,end)
                if not self._inference:
                    return batch_data , batch_labels
                else:
                    return batch_data
            elif self._return_type == 1:
                return self.get_batch_data(self._data_s1,start,end), self.get_batch_data(self._labels, start, end)
            elif self._return_type == 2:
                return self.get_batch_data(self._data_s2,start,end), self.get_batch_data(self._labels, start, end)
