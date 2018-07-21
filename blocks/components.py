import os
import numpy as np
import tensorflow as tf

from tensorflow.contrib.framework.python.ops import arg_scope, add_arg_scope
from blocks.helpers import int_shape, get_name
from blocks.layers import conv2d, deconv2d, dense, nin, gated_resnet

@add_arg_scope
def omniglot_conv_encoder(inputs, r_dim, is_training, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, counters={}):
    name = get_name("omniglot_conv_encoder", counters)
    print("construct", name, "...")
    with tf.variable_scope(name):
        with arg_scope([conv2d, dense], nonlinearity=nonlinearity, bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training):
            outputs = inputs
            outputs = conv2d(outputs, 64, 3, 1, "SAME")
            outputs = conv2d(outputs, 64, 3, 2, "SAME")
            outputs = conv2d(outputs, 128, 3, 1, "SAME")
            outputs = conv2d(outputs, 128, 3, 2, "SAME")
            outputs = conv2d(outputs, 256, 4, 1, "VALID")
            outputs = conv2d(outputs, 256, 4, 1, "VALID")
            outputs = tf.reshape(outputs, [-1, 256])
            r = tf.dense(outputs, r_dim, nonlinearity=None, bn=False)
            return r


@add_arg_scope
def fc_encoder(X, y, r_dim, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, is_training=False, counters={}):
    inputs = tf.concat([X, y[:, None]], axis=1)
    name = get_name("fc_encoder", counters)
    print("construct", name, "...")
    with tf.variable_scope(name):
        with arg_scope([dense], nonlinearity=nonlinearity, bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training):
            outputs = dense(inputs, 256)
            outputs = nonlinearity(dense(outputs, 256, nonlinearity=None) + dense(inputs, 256, nonlinearity=None))
            outputs = dense(outputs, 256)
            outputs = nonlinearity(dense(outputs, 256, nonlinearity=None) + dense(inputs, 256, nonlinearity=None))
            outputs = dense(outputs, r_dim, nonlinearity=None, bn=False)
            return outputs

@add_arg_scope
def aggregator(r, num_c, z_dim, method=tf.reduce_mean, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, is_training=False, counters={}):
    name = get_name("aggregator", counters)
    print("construct", name, "...")
    with tf.variable_scope(name):
        with arg_scope([dense], nonlinearity=nonlinearity, bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training):
            r_pr = method(r[:num_c], axis=0, keepdims=True)
            r = method(r, axis=0, keepdims=True)
            r = tf.concat([r_pr, r], axis=0)
            r = dense(r, 256)
            r = dense(r, 256)
            r = dense(r, 256)
            z_mu = dense(r, z_dim, nonlinearity=None, bn=False)
            z_log_sigma_sq = dense(r, z_dim, nonlinearity=None, bn=False)
            return z_mu[:1], z_log_sigma_sq[:1], z_mu[1:], z_log_sigma_sq[1:]

@add_arg_scope
def conditional_decoder(x, z, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, is_training=False, counters={}):
    name = get_name("conditional_decoder", counters)
    print("construct", name, "...")
    with tf.variable_scope(name):
        with arg_scope([dense], nonlinearity=nonlinearity, bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training):
            batch_size = tf.shape(x)[0]
            x = tf.tile(x, tf.stack([1, int_shape(z)[1]]))
            z = tf.tile(z, tf.stack([batch_size, 1]))
            xz = x + z * tf.get_variable(name="coeff", shape=(), dtype=tf.float32, initializer=tf.constant_initializer(2.0))
            a = dense(xz, 256, nonlinearity=None) + dense(z, 256, nonlinearity=None)
            outputs = tf.nn.tanh(a) * tf.sigmoid(a)

            for k in range(4):
                a = dense(outputs, 256, nonlinearity=None) + dense(z, 256, nonlinearity=None)
                outputs = tf.nn.tanh(a) * tf.sigmoid(a)
            outputs = dense(outputs, 1, nonlinearity=None, bn=False)
            outputs = tf.reshape(outputs, shape=(batch_size,))
            return outputs

            # x = tf.tile(x, tf.stack([1, int_shape(z)[1]]))
            # batch_size = tf.shape(x)[0]
            # z = tf.tile(z, tf.stack([batch_size, 1]))
            # xz = x + z #* tf.get_variable(name="coeff", shape=(), dtype=tf.float32, initializer=tf.constant_initializer(2.0))
            # outputs = dense(xz, 512)
            # outputs = dense(outputs, 512)
            # outputs = dense(outputs, 512)
            # outputs = dense(outputs, 512)
            # outputs = dense(outputs, 512)
            # outputs = dense(outputs, 1, nonlinearity=None, bn=False)
            # outputs = tf.reshape(outputs, shape=(batch_size,))
            # return outputs


            # batch_size = tf.shape(x)[0]
            # z = tf.tile(z, tf.stack([batch_size, 1]))
            #
            # outputs = dense(z, 256) + x
            # outputs = dense(outputs, 256) + x
            # outputs = dense(outputs, 256) + x
            # outputs = dense(outputs, 256) + x
            # outputs = dense(outputs, 1, nonlinearity=None, bn=False)
            # outputs = tf.reshape(outputs, shape=(batch_size,))
            # return outputs


            # batch_size = tf.shape(x)[0]
            # z = tf.tile(z, tf.stack([batch_size, 1]))
            # xz = tf.concat([x, z], axis=1)
            # outputs = dense(xz, 512)
            # outputs = dense(outputs, 512)
            # outputs = dense(outputs, 512)
            # outputs = dense(outputs, 512)
            # outputs = dense(outputs, 512)
            # outputs = dense(outputs, 1, nonlinearity=None, bn=False)
            # outputs = tf.reshape(outputs, shape=(batch_size,))
            # return outputs







# @add_arg_scope
# def fc_encoder(inputs, r_dim, is_training, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, counters={}):
#     name = get_name("fc_encoder", counters)
#     print("construct", name, "...")
#     with tf.variable_scope(name):
#         with arg_scope([dense], nonlinearity=nonlinearity, bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training):
#             outputs = dense(inputs, 10)
#             outputs = dense(outputs, 10)
#             outputs = dense(outputs, r_dim, nonlinearity=None)
#             return outputs
#
# @add_arg_scope
# def aggregator(r, z_dim, method=tf.reduce_mean, counters={}):
#     name = get_name("aggregator", counters)
#     print("construct", name, "...")
#     with tf.variable_scope(name):
#         r = method(r, axis=0, keepdims=True)
#         z_mu = dense(r, z_dim, nonlinearity=None, bn=False)
#         z_log_sigma_sq = dense(r, z_dim, nonlinearity=None, bn=False)
#         return z_mu, z_log_sigma_sq
#
# @add_arg_scope
# def conditional_decoder(x, z, is_training, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, counters={}):
#     name = get_name("conditional_decoder", counters)
#     print("construct", name, "...")
#     with tf.variable_scope(name):
#         with arg_scope([dense], nonlinearity=nonlinearity, bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training):
#             batch_size = tf.shape(x)[0]
#             z = tf.tile(z, tf.stack([batch_size, 1]))
#             xz = tf.concat([x, z], axis=1)
#             outputs = dense(xz, 50)
#             outputs = dense(outputs, 50)
#             outputs = dense(outputs, 50)
#             outputs = dense(outputs, 1, nonlinearity=None)
#             outputs = tf.reshape(outputs, shape=(batch_size,))
#             return outputs
