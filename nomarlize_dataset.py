from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def add_image(image,sum,s_var):
    for row in image:
        for i in row:
            sum = sum + i
            s_var = s_var + i*i
    return sum, s_var

def get_avg_var(data):

    # initilize the layer_sum:
    layer_sum = []
    layer_s_var = []
    for i in range(0,data.shape[3]):
        layer_sum.append(0)
        layer_s_var.append(0)

    for per_data in data:
        for channel_num in range(0,per_data.shape[2]):
            layer_sum[channel_num], layer_s_var[channel_num] = \
                add_image(per_data[:,:,channel_num],layer_sum[channel_num], layer_s_var[channel_num])

    layer_avg , layer_var = [],[]
    sum_var = []
    for i in range(0, data.shape[3]):
        layer_avg.append(layer_sum[i] / data.shape[0])
        layer_var.append((layer_s_var[i] - layer_avg[i]*layer_avg[i]) / (data.shape[0]-1))
        sum_var.append(layer_s_var[i]/ (data.shape[0]-1))

    print('sum_var:',sum_var)

    return layer_avg , layer_var






'''from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import h5py
import os
from nomarlize_dataset import get_avg_var

### to change according to your machine
base_dir = os.path.expanduser('D:\Documents\script\python_script\AI\competation\dataset')
# path_training = os.path.join(base_dir, 'training.h5')
path_validation = os.path.join(base_dir, 'validation.h5')

# print(path_training)
#fid_training = h5py.File(path_training,'r')
fid_validation = h5py.File(path_validation,'r')

print(fid_validation['sen1'].shape)

avg,var = get_avg_var(fid_validation['sen1'])


print('average for each channel is ',avg)
print('var for each channel is ' ,var)'''