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
from blocks.components import omniglot_conv_encoder


class BinaryPixelCNN(object):

    def __init__(self, counters={}):
        self.counters = counters

    def construct(self, inputs, is_training, dropout_p, nr_resnet=1, nr_filters=50, nonlinearity=tf.nn.relu, bn=False, kernel_initializer=None, kernel_regularizer=None):
        self.inputs = inputs
        self.nr_filters = nr_filters
        self.nonlinearity = nonlinearity
        self.dropout_p = dropout_p
        self.bn = bn
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.is_training = is_training

        self.outputs = self._model(inputs, nr_resnet, nr_filters, nonlinearity, dropout_p, bn, kernel_initializer, kernel_regularizer, is_training)
        self.loss = self._loss(self.inputs, self.outputs)

        self.x_hat = bernoulli_sampler(self.outputs)


    def _model(self, x, nr_resnet, nr_filters, nonlinearity, dropout_p, bn, kernel_initializer, kernel_regularizer, is_training):
        pass

    def _loss(self, x, outputs):
        l =  tf.reduce_mean(bernoulli_loss(x, outputs, sum_all=False))
        return l
