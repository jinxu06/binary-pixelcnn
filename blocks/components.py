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
            # z_mu = dense(outputs, z_dim, nonlinearity=None, bn=False)
            # z_log_sigma_sq = dense(outputs, z_dim, nonlinearity=None, bn=False)
            # return z_mu, z_log_sigma_sq
