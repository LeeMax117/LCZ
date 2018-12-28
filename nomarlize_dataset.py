from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

def add_image(image,sum,s_var):
    for i in image:
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
            layer_sum[channel_num], layer_s_var[channel_num] = add_image(per_data[:,:,channel_num])
