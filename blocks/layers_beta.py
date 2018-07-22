"""
Various tensorflow utilities
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope
from blocks.helpers import int_shape, get_name


@add_arg_scope
def dense(inputs, num_outputs, W=None, b=None, nonlinearity=None, bn=False, kernel_initializer=None, kernel_regularizer=None, is_training=False, counters={}):
    ''' fully connected layer '''
    name = get_name('dense', counters)
    with tf.variable_scope(name):
        if W is None:
            W = tf.get_variable('W', shape=[int(inputs.get_shape()[1]),num_outputs], dtype=tf.float32, trainable=True, initializer=kernel_initializer, regularizer=kernel_regularizer)
        if b is None:
            b = tf.get_variable('b', shape=[num_outputs], dtype=tf.float32, trainable=True, initializer=tf.constant_initializer(0.), regularizer=None)

        outputs = tf.matmul(inputs, W) + tf.reshape(b, [1, num_outputs])

        if bn:
            outputs = tf.layers.batch_normalization(outputs, training=is_training)
        if nonlinearity is not None:
            outputs = nonlinearity(outputs)
        print("    + dense", int_shape(inputs), int_shape(outputs), nonlinearity, bn)
        return outputs

# @add_arg_scope
# def conv2d(x, num_filters, filter_size=[3,3], stride=[1,1], pad='SAME', nonlinearity=None, init_scale=1., counters={}, init=False, ema=None, **kwargs):
#     ''' convolutional layer '''
#     name = get_name('conv2d', counters)
#     with tf.variable_scope(name):
#         V = get_var_maybe_avg('V', ema, shape=filter_size+[int(x.get_shape()[-1]),num_filters], dtype=tf.float32,
#                               initializer=tf.random_normal_initializer(0, 0.05), trainable=True)
#         g = get_var_maybe_avg('g', ema, shape=[num_filters], dtype=tf.float32,
#                               initializer=tf.constant_initializer(1.), trainable=True)
#         b = get_var_maybe_avg('b', ema, shape=[num_filters], dtype=tf.float32,
#                               initializer=tf.constant_initializer(0.), trainable=True)
#
#         # use weight normalization (Salimans & Kingma, 2016)
#         W = tf.reshape(g, [1, 1, 1, num_filters]) * tf.nn.l2_normalize(V, [0, 1, 2])
#
#         # calculate convolutional layer output
#         x = tf.nn.bias_add(tf.nn.conv2d(x, W, [1] + stride + [1], pad), b)
#
#         if init:  # normalize x
#             m_init, v_init = tf.nn.moments(x, [0,1,2])
#             scale_init = init_scale / tf.sqrt(v_init + 1e-10)
#             with tf.control_dependencies([g.assign(g * scale_init), b.assign_add(-m_init * scale_init)]):
#                 x = tf.identity(x)
#
#         # apply nonlinearity
#         if nonlinearity is not None:
#             x = nonlinearity(x)
#
#         return x
#
# @add_arg_scope
# def deconv2d(x, num_filters, filter_size=[3,3], stride=[1,1], pad='SAME', nonlinearity=None, init_scale=1., counters={}, init=False, ema=None, **kwargs):
#     ''' transposed convolutional layer '''
#     name = get_name('deconv2d', counters)
#     xs = int_shape(x)
#     if pad=='SAME':
#         target_shape = [xs[0], xs[1]*stride[0], xs[2]*stride[1], num_filters]
#     else:
#         target_shape = [xs[0], xs[1]*stride[0] + filter_size[0]-1, xs[2]*stride[1] + filter_size[1]-1, num_filters]
#     with tf.variable_scope(name):
#         V = get_var_maybe_avg('V', ema, shape=filter_size+[num_filters,int(x.get_shape()[-1])], dtype=tf.float32,
#                               initializer=tf.random_normal_initializer(0, 0.05), trainable=True)
#         g = get_var_maybe_avg('g', ema, shape=[num_filters], dtype=tf.float32,
#                               initializer=tf.constant_initializer(1.), trainable=True)
#         b = get_var_maybe_avg('b', ema, shape=[num_filters], dtype=tf.float32,
#                               initializer=tf.constant_initializer(0.), trainable=True)
#
#         # use weight normalization (Salimans & Kingma, 2016)
#         W = tf.reshape(g, [1, 1, num_filters, 1]) * tf.nn.l2_normalize(V, [0, 1, 3])
#
#         # calculate convolutional layer output
#         x = tf.nn.conv2d_transpose(x, W, target_shape, [1] + stride + [1], padding=pad)
#         x = tf.nn.bias_add(x, b)
#
#         if init:  # normalize x
#             m_init, v_init = tf.nn.moments(x, [0,1,2])
#             scale_init = init_scale / tf.sqrt(v_init + 1e-10)
#             with tf.control_dependencies([g.assign(g * scale_init), b.assign_add(-m_init * scale_init)]):
#                 x = tf.identity(x)
#
#         # apply nonlinearity
#         if nonlinearity is not None:
#             x = nonlinearity(x)
#
#         return x
