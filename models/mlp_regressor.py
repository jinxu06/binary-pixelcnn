import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope, add_arg_scope
from blocks.layers import conv2d, deconv2d, dense, nin, gated_resnet
from blocks.layers import up_shifted_conv2d, up_left_shifted_conv2d, up_shift, left_shift
from blocks.layers import down_shifted_conv2d, down_right_shifted_conv2d, down_shift, right_shift, down_shifted_deconv2d, down_right_shifted_deconv2d
from blocks.losses import bernoulli_loss
from blocks.samplers import gaussian_sampler, mix_logistic_sampler, bernoulli_sampler
from blocks.helpers import int_shape, broadcast_masks_tf

class MLPRegressor(object):

    def __init__(self, counters={}):
        self.counters = counters

    def construct(self, X, y, is_training, nonlinearity=tf.nn.relu, bn=False, kernel_initializer=None, kernel_regularizer=None):
        self.X = X
        self.y = y
        self.nonlinearity = nonlinearity
        self.bn = bn
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.is_training = is_training

        self.outputs = self._model(X, nonlinearity, bn, kernel_initializer, kernel_regularizer, is_training)
        self.predictions = self.outputs
        self.loss = self._loss(self.y, self.predictions)

    def _model(self, x, nonlinearity, bn, kernel_initializer, kernel_regularizer, is_training):
        bsize = int_shape(x)
        with arg_scope([dense], nonlinearity=nonlinearity, bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training, counters=self.counters):
            outputs = dense(x, 100)
            outputs = dense(outputs, 100)
            outputs = dense(outputs, 100)
            outputs = dense(outputs, 1, nonlinearity=None)
            outputs = tf.reshape(outputs, shape=(bsize[0],))
            return outputs


    def _loss(self, y, predictions):
        return tf.losses.mean_squared_error(labels=y, predictions=predictions)
