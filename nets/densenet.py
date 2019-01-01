"""Contains a variant of the densenet model definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def trunc_normal(stddev): return tf.truncated_normal_initializer(stddev=stddev)


def bn_act_conv_drp(current, num_outputs, kernel_size, scope='block'):
    current = slim.batch_norm(current, scope=scope + '_bn')
    current = tf.nn.relu(current)
    current = slim.conv2d(current, num_outputs, kernel_size, scope=scope + '_conv')
    current = slim.dropout(current, scope=scope + '_dropout')
    return current


def block(net, layers, growth, scope='block'):
    for idx in range(layers):
        bottleneck = bn_act_conv_drp(net, 4 * growth, [1, 1],
                                     scope=scope + '_conv1x1' + str(idx))
        tmp = bn_act_conv_drp(bottleneck, growth, [3, 3],
                              scope=scope + '_conv3x3' + str(idx))
        net = tf.concat(axis=3, values=[net, tmp])
    return net


def densenet(images, num_classes=1001, is_training=False,
             dropout_keep_prob=0.8,
             scope='densenet'):
    """Creates a variant of the densenet model.

      images: A batch of `Tensors` of size [batch_size, height, width, channels].
      num_classes: the number of classes in the dataset.
      is_training: specifies whether or not we're currently training the model.
        This variable will determine the behaviour of the dropout layer.
      dropout_keep_prob: the percentage of activation values that are retained.
      prediction_fn: a function to get predictions out of logits.
      scope: Optional variable_scope.

    Returns:
      logits: the pre-softmax activations, a tensor of size
        [batch_size, `num_classes`]
      end_points: a dictionary from components of the network to the corresponding
        activation.
    """
    growth = 24
    compression_rate = 0.5

    def reduce_dim(input_feature):
        return int(int(input_feature.shape[-1]) * compression_rate)

    end_points = {}

    with tf.variable_scope(scope, 'DenseNet', [images, num_classes]):
        with slim.arg_scope(bn_drp_scope(is_training=is_training,
                                         keep_prob=dropout_keep_prob)) as ssc:
            # 32 x 32 x 18
            # first conv layer with conv kernal 7*7*2growth
            # use concat like inception
            scope = 'conv1_a'
            branch_0 = slim.conv2d(images, growth,[7,7],scope=scope)
            end_points[scope] = branch_0

            scope = 'conv1_b'
            branch_1 = slim.conv2d(images, growth,[3,3],scope=scope)
            end_points[scope] = branch_1

            scope = 'mixed_conv1'
            net = tf.concat(axis=3, values=[branch_0, branch_1])
            end_points[scope] = net

            '''
            # first pooling layer with max pooling and stride is 2
            scope = 'pool1'
            net = slim.max_pool2d(net,[3,3],stride = 2,scope=scope)
            end_points[scope] = net
            '''
            # 32 x 32 x 2 growth
            # first dense block with 6 layer which will increase 6 growth feature map
            scope = 'block1'
            net = block(net,6,growth,scope=scope)
            end_points[scope] = net

            # 32 x 32 x 8 growth
            # first transition Layer
            scope = 'transition1'
            # the 1*1 conv in transition layer
            net = bn_act_conv_drp(net,reduce_dim(net),[1,1],scope = scope)
            end_points[scope] = net
            # the avg pooling in transition layer
            scope = 'avgpool1'
            net = slim.avg_pool2d(net,[2,2],stride = 2,scope=scope)
            end_points[scope] = net
            # output layers: 4 growth

            # sencond dense block
            scope = 'block2'
            net = block(net,12,growth,scope=scope)
            end_points[scope] = net
            # output layers: 4 + 12 growth

            scope = 'transition2'
            net = bn_act_conv_drp(net,reduce_dim(net),[1,1],scope = scope)
            end_points[scope] = net
            # the avg pooling in transition layer
            scope = 'avgpool2'
            net = slim.avg_pool2d(net,[2,2],stride = 2,scope=scope)
            end_points[scope] = net
            # output layers: 8 growth

            # third dense block
            scope = 'block3'
            net = block(net,32,growth,scope=scope)
            end_points[scope] = net
            # output layers: 8+32 growth

            # third transition layers:
            scope = 'transition3'
            net = bn_act_conv_drp(net,reduce_dim(net),[1,1],scope = scope)
            # the avg pooling in transition layer
            scope = 'avgpool3'
            net = slim.avg_pool2d(net,[2,2],stride = 2,scope=scope)
            end_points[scope] = net
            # output layers: 20 growth

            # 4th dense block
            scope = 'block4'
            net = block(net,32,growth,scope=scope)
            end_points[scope] = net
            # output layers: 20 + 32 growth

            # batch_normal
            scope = 'last_batch_norm_relu'
            net = slim.batch_norm(net,scope=scope)
            net = tf.nn.relu(net)

            # As the input image size is unknow
            # use a average pooling to set the feature map size to 1*1
            scope = 'global_average'
            net = slim.avg_pool2d(net,net.shape[1:3],scope=scope)

            # last layer to classfier
            # use a 1*1 conv to generate pre_logits
            biases_initializer = tf.constant_initializer(0.1)
            scope = 'pre_logits'
            pre_logit = slim.conv2d(net,num_classes,[1,1],biases_initializer=biases_initializer,
                    scope = scope)
            end_points[scope] = net

            scope = 'logits'
            logits = tf.squeeze(pre_logit)
            end_points[scope] = logits
            print(logits)

    return logits, end_points


def bn_drp_scope(is_training=True, keep_prob=0.8):
    keep_prob = keep_prob if is_training else 1
    with slim.arg_scope(
        [slim.batch_norm],
            scale=True, is_training=is_training, updates_collections=None):
        with slim.arg_scope(
            [slim.dropout],
                is_training=is_training, keep_prob=keep_prob) as bsc:
            return bsc


def densenet_arg_scope(weight_decay=0.004):
    """Defines the default densenet argument scope.

    Args:
      weight_decay: The weight decay to use for regularizing the model.

    Returns:
      An `arg_scope` to use for the inception v3 model.
    """
    with slim.arg_scope(
        [slim.conv2d],
        weights_initializer=tf.contrib.layers.variance_scaling_initializer(
            factor=2.0, mode='FAN_IN', uniform=False),
        activation_fn=None, biases_initializer=None, padding='same',
            stride=1) as sc:
        return sc


densenet.default_image_size = 224
