# Brief:     Build densnet graph
# Data:      28/Aug./2017
# E-mail:    huyixuanhyx@gmail.com
# License:   Apache 2.0
# By:        Yeephycho @ Hong Kong

# Code still under construction


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np



DATA_DIR = "./tfrecord"
TRAINING_SET_SIZE = 2512
global_step = TRAINING_SET_SIZE * 100
TEST_SET_SIZE = 908
BATCH_SIZE = 16
IMAGE_SIZE = 224



def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.05)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.02, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')



def batch_norm(input_tensor, if_training):
    """
    Batch normalization on convolutional feature maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        input_tensor:           Tensor, 4D NHWC input feature maps
        depth:                  Integer, depth of input feature maps
        if_training:            Boolean tf.Varialbe, true indicates training phase
        scope:                  String, variable scope
    Return:
        normed_tensor:          Batch-normalized feature maps
    """
    with tf.variable_scope('batch_normalization'):
        depth = int(input_tensor.get_shape()[-1])
        beta = tf.Variable(tf.constant(0.0, shape=[depth]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[depth]),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(input_tensor, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.99)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(if_training,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed_tensor = tf.nn.batch_normalization(input_tensor, mean, var, beta, gamma, 1e-3)
    return normed_tensor



def composite_function(__input_tensor, __input_tensor_depth, __output_tensor_depth, if_training):
    __conv_weights = weight_variable([3, 3, __input_tensor_depth, __output_tensor_depth])
    __output_tensor = batch_norm(__input_tensor, if_training)
    __output_tensor = tf.nn.relu(__output_tensor)
    __output_tensor = tf.nn.conv2d(input=__output_tensor, filter=__conv_weights, strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC', name='composite_3x3_s1')

    return __output_tensor


def transition_layer(__input_tensor, __input_tensor_depth, __output_tensor_depth, if_training):
    __conv_weights = weight_variable([1, 1, __input_tensor_depth, __output_tensor_depth])
    __output_tensor = batch_norm(__input_tensor, if_training)
    __output_tensor = tf.nn.conv2d(input=__output_tensor, filter=__conv_weights, strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC', name='transition_1x1_s1')
    __output_tensor = tf.nn.avg_pool(value=__output_tensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', data_format='NHWC', name='avg_pool_2x2')

    return __output_tensor


def bottleneck_layer(_input_tensor, _input_tensor_depth, _output_tensor_depth, if_training):
    __conv_weights = weight_variable([1, 1, _input_tensor_depth, _output_tensor_depth]) #NOTE: output_tensor should be 4k, 4 times of the growth_rate
    __output_tensor = batch_norm(_input_tensor, if_training)
    __output_tensor = tf.nn.relu(__output_tensor)
    __output_tensor = tf.nn.conv2d(input=__output_tensor, filter=__conv_weights, strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC', name='bottleneck_1x1_s1')
    __output_tensor = composite_function(__output_tensor, 128, 32, if_training) #NOTE: output/input_tensor_depth is different from input

    return __output_tensor



def dense_block(_input_tensor, growth_rate, if_training):
    _bottleneck_output_0 = bottleneck_layer(_input_tensor, 64, 128, if_training)#NOTE:64 is 2k, here k = 32, 128 is 4k, output is k = 32
    _bottleneck_input_0 = tf.concat(values=[_input_tensor, _bottleneck_output_0], axis=3, name='stack0')# 96

    _bottlenect_output_1 = bottleneck_layer(_bottleneck_input_0, 96, 128, if_training)#NOTE:96 = 64 + 32
    _bottleneck_input_1 = tf.concat(values=[_input_tensor, _bottleneck_output_0, _bottlenect_output_1], axis=3, name='stack1')# 128

    _bottlenect_output_2 = bottleneck_layer(_bottleneck_input_1, 128, 128, if_training)#NOTE:128 = 96 + 32
    _bottleneck_input_2 = tf.concat(values=[_input_tensor, _bottleneck_output_0, _bottlenect_output_1, _bottlenect_output_2], axis=3, name='stack2')# 160

    _bottlenect_output_3 = bottleneck_layer(_bottleneck_input_2, 160, 128, if_training)#NOTE:160 = 128 + 32
    _bottleneck_input_3 = tf.concat(values=[_input_tensor, _bottleneck_output_0, _bottlenect_output_1, _bottlenect_output_2, _bottlenect_output_3], axis=3, name='stack3')# 192

    _bottlenect_output_4 = bottleneck_layer(_bottleneck_input_3, 192, 128, if_training)#NOTE:192 = 160 + 32
    _bottleneck_input_4 = tf.concat(values=[_input_tensor, _bottleneck_output_0, _bottlenect_output_1, _bottlenect_output_2, _bottlenect_output_3, _bottlenect_output_4], axis=3, name='stack4')# 224

    output_tensor = bottleneck_layer(_bottleneck_input_4, 224, 128, if_training)#NOTE:192 = 160 + 32

    return output_tensor


def densenet_inference(image_batch, if_training, dropout_prob):
    with tf.name_scope('densenet_forward_graph'):
        _image_batch = tf.reshape(image_batch, [-1, IMAGE_SIZE, IMAGE_SIZE, 3])

        with tf.name_scope('conv2d_7x7_s2') as scope:
            _conv_weights = weight_variable([7, 7, 3, 64])
            _output_tensor = tf.nn.conv2d(input=_image_batch, filter=_conv_weights, strides=[1, 2, 2, 1], padding='SAME', data_format='NHWC', name='conv2d_7x7_s2')
            _output_tensor = tf.nn.max_pool(value=_output_tensor, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', data_format='NHWC', name='max_pool_3x3_s2')
            _output_tensor = tf.nn.relu(_output_tensor)

        # with tf.name_scope('conv2'):
        #     W_conv2 = weight_variable([5, 5, 64, 64])
        #     b_conv2 = bias_variable([64])
        #
        #     h_conv2 = tf.nn.relu(conv2d(_output_tensor, W_conv2) + b_conv2)
        #     h_pool2 = max_pool_2x2(h_conv2) # 56

        with tf.name_scope('dense_block_0') as scope:
            _output_tensor = dense_block(_output_tensor, 32, if_training)


        with tf.name_scope('conv3'):
            W_conv3 = weight_variable([5, 5, 32, 128])
            b_conv3 = bias_variable([128])

            h_conv3 = tf.nn.relu(conv2d(_output_tensor, W_conv3) + b_conv3)
            h_pool3 = max_pool_2x2(h_conv3) # 28

        with tf.name_scope('conv4'):
            W_conv4 = weight_variable([3, 3, 128, 256])
            b_conv4 = bias_variable([256])

            h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
            h_pool4 = max_pool_2x2(h_conv4) # 14

        with tf.name_scope('conv5'):
            W_conv5 = weight_variable([3, 3, 256, 256])
            b_conv5 = bias_variable([256])

            h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
            h_pool5 = max_pool_2x2(h_conv5) # 7

        with tf.name_scope('fc'):
            W_fc1 = weight_variable([7*7*256, 1024])
            b_fc1 = bias_variable([1024])

            h_pool5_flat = tf.reshape(h_pool5, [-1, 7*7*256])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat, W_fc1) + b_fc1)

            h_fc1_drop = tf.nn.dropout(h_fc1, dropout_prob)

            W_fc2 = weight_variable([1024, 256])
            b_fc2 = bias_variable([256])

            h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

            W_fc3 = weight_variable([256, 64])
            b_fc3 = bias_variable([64])

            h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)

            W_fc4 = weight_variable([64, 5])
            b_fc4 = bias_variable([5])

            # y_conv = tf.nn.softmax(tf.matmul(h_fc3, W_fc4) + b_fc4)
            y_conv = tf.matmul(h_fc3, W_fc4) + b_fc4

    return y_conv


# # from cifar10 example @ https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10.py
# def _variable_on_gpu(name, shape, initializer):
#   """Helper to create a Variable stored on CPU memory.
#   Args:
#     name: name of the variable
#     shape: list of ints
#     initializer: initializer for Variable
#   Returns:
#     Variable Tensor
#   """
#   with tf.device('/gpu:0'):
#     dtype = tf.float32
#     var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
#   return var
#
# def _variable_with_weight_decay(name, shape, stddev, wd):
#   """Helper to create an initialized Variable with weight decay.
#   Note that the Variable is initialized with a truncated normal distribution.
#   A weight decay is added only if one is specified.
#   Args:
#     name: name of the variable
#     shape: list of ints
#     stddev: standard deviation of a truncated Gaussian
#     wd: add L2Loss weight decay multiplied by this float. If None, weight
#         decay is not added for this Variable.
#   Returns:
#     Variable Tensor
#   """
#   dtype = tf.float32
#   var = _variable_on_gpu(
#       name,
#       shape,
#       tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
#   if wd is not None:
#     weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
#     tf.add_to_collection('losses', weight_decay)
#   return var
