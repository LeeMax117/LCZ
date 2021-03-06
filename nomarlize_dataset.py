from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import numpy as np

def add_batch_data(data,sum,s_var):
    sum = sum + data.sum(axis = 0).sum(axis=0).sum(axis=0)
    for i in range(len(s_var)):
        s_var[i] = s_var[i] + np.sum(np.square(data[:,:,:,i]))
    print('a batch of calculate done')
    return sum, s_var

def get_avg_stand(data,batch = 100):
    batch_size = batch
    num = data.shape[0]
    used_num = 0
    layer_sum = []
    layer_s_var = []
    for i in range(0, data.shape[3]):
        layer_sum.append(0)
        layer_s_var.append(0)
    while used_num < num:
        batch_data = np.array(data[used_num:used_num + batch_size, :, :, :])
        used_num += batch_size
        layer_sum ,layer_s_var = add_batch_data(batch_data,layer_sum,layer_s_var)

    layer_avg, layer_var, layer_standard = [], [], []
    total_num = data.shape[0] * data.shape[1] * data.shape[2]
    for i in range(0, data.shape[3]):
        layer_avg.append(layer_sum[i] / total_num)
        layer_var.append((layer_s_var[i] - total_num * layer_avg[i] * layer_avg[i]) / (total_num - 1))
        layer_standard.append(math.sqrt(layer_var[i]))

    return layer_avg, layer_standard

'''
def add_image(image, sum_, s_var):
    for row in image:
        for i in row:
            sum_ = sum_ + i
            s_var = s_var + i*i
    return sum_, s_var

def get_avg_standard(data):
    # initilize the layer_sum:
    layer_sum = []
    layer_s_var = []
    for i in range(0, data.shape[3]):
        layer_sum.append(0)
        layer_s_var.append(0)

    for per_data in data:
        for channel_num in range(0, per_data.shape[2]):
            layer_sum[channel_num], layer_s_var[channel_num] = \
                add_image(per_data[:, :, channel_num], layer_sum[channel_num], layer_s_var[channel_num])

    layer_avg, layer_var, layer_standard = [], [], []
    sum_var = []
    total_num = data.shape[0]*data.shape[1]*data.shape[2]
    for i in range(0, data.shape[3]):
        layer_avg.append(layer_sum[i] / total_num)
        layer_var.append((layer_s_var[i] - total_num * layer_avg[i]*layer_avg[i]) / (total_num-1))
        layer_standard.append(math.sqrt(layer_var[i]))

    return layer_avg, layer_standard
'''

if __name__ == '__main__':
    n = np.ones((5, 2, 2, 5))
    test = np.array([2,2,2,2],[3,3,3,3],)
    # n = n.reshape(2, 2, 2, 4)
    # print(n)
    print(get_avg_stand(n))

'''
    import h5py
    import os

    ### to change according to your machine
    base_dir = os.path.expanduser('D:\Documents\script\python_script\AI\competation\dataset')
    # path_training = os.path.join(base_dir, 'training.h5')
    path_validation = os.path.join(base_dir, 'validation.h5')

    # print(path_training)
    # fid_training = h5py.File(path_training,'r')
    fid_validation = h5py.File(path_validation, 'r')

    print(fid_validation['sen1'].shape)

    avg, var = get_avg_stand(fid_validation['sen1'],1000)

    print('average for each channel is ', avg)
    print('var for each channel is ', var)
'''