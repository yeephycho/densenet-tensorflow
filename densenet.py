# Brief:     Train a densenet for image classification
# Data:      24/Aug./2017
# E-mail:    huyixuanhyx@gmail.com
# License:   Apache 2.0
# By:        Yeephycho @ Hong Kong

# Code still under construction


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


growth_rate = 12

def batch_norm(input_tensor, if_training=True):
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
        depth = int(tf.shape(input_tensor)[-1])
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


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def conv2d(input_tensor, output_depth, kernel_size):
    input_depth = int(tf.shape(input_tensor)[-1])
    kernel = weight_variable(
        [kernel_size, kernel_size, input_depth, output_depth],
        name='kernel')
    output_tensor = tf.nn.conv2d(input=input_tensor, filter=kernel, strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC', name='conv2d')
    return output_tensor


def conv2d_3x3(input_tensor, output_depth):
    with tf.variable_scope('conv2d_3x3'):
        return conv2d(input_tensor, output_depth, 3) #TODO: set the output_depth according to the originial paper.


def conv2d_1x1(input_tensor, output_depth):
    with tf.variable_scope('conv2d_1x1'):
        return conv2d(input_tensor, output_depth, 1) #TODO: set the output_depth according to the original paper.


def drop_out(input_tensor, if_training):
    with tf.varialbe_scop('drop_out'):
        if keep_prob < 1:                          #TODO: set keep_prob for drop out as a hyper parameter.
            output_tensor = tf.cond(
                if_training,
                lambda: tf.nn.dropout(input_tensor, keep_prob),
                lambda: input_tensor
            )
        else:
            output_tensor = input_tensor
        return output_tensor


def composite_function(input_tensor, if_training=True):
    normed_tensor = batch_norm(input_tensor, if_training=if_training)
    actived_tensor = tf.nn.relu(normed_tensor, name='relu')
    output_tensor = conv2d_3x3(actived_tensor, output_depth)
    output_tensor = drop_out(input_tensor, if_training)
    return output_tensor


def avg_pool_2x2(input_tensor):
    with tf.variable_scope('avg_pool_2x2'):
        return tf.nn.avg_pool(value=input_tensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', data_format='NHWC', name='avg_pool_2x2')


def transition_layer(input_tensor, if_training=True):
    normed_tensor = batch_norm(input_tensor, if_training=if_training)
    conv_tensor = conv2d_1x1(normed_tensor, output_depth)
    output_tensor = avg_pool_2x2(input_tensor)
    return output_tensor


def bottleneck_layer(input_tensor, if_training):
    with tf.variable_scope('avg_pool_2x2'):
        normed_tensor = batch_norm(input_tensor, if_training=if_training)
        actived_tensor = tf.nn.relu(normed_tensor, name='relu')
        conv_tensor_a = conv2d_1x1(actived_tensor, 4 * growth_rate) #NOTE: according to the paper, 4 * k reduction dimension output depth
        normed_tensor = batch_norm(conv_tensor_a, if_training=if_training)
        actived_tensor = tf.nn.relu(normed_tensor, name='relu')
        conv_tensor_a = conv2d_3x3(actived_tensor, growth_rate)
