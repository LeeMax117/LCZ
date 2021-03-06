# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains a factory for building various models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools

import tensorflow as tf

from nets import inception
from nets import densenet

slim = tf.contrib.slim

networks_map = {
                'inception_v4': inception.inception_v4,
                'M_inception_v4': inception.M_inception_v4,
                'densenet': densenet.densenet,
               }

arg_scopes_map = {
                  'inception_v4': inception.inception_v4_arg_scope,
                  'M_inception_v4':inception.inception_v4_arg_scope,
                  'densenet': densenet.densenet_arg_scope,
                 }


def get_network_fn(name, num_classes, weight_decay=0.0, is_training=False):
  if name not in networks_map:
    raise ValueError('Name of network unknown %s' % name)
  func = networks_map[name]
  @functools.wraps(func)
  def network_fn(images, **kwargs):
    arg_scope = arg_scopes_map[name](weight_decay=weight_decay)
    with slim.arg_scope(arg_scope):
      return func(images, num_classes, is_training=is_training, **kwargs)
  if hasattr(func, 'default_image_size'):
    network_fn.default_image_size = func.default_image_size

  return network_fn
