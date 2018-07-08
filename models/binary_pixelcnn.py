import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope, add_arg_scope
from blocks.layers import conv2d, deconv2d, dense, nin, gated_resnet
from blocks.layers import up_shifted_conv2d, up_left_shifted_conv2d, up_shift, left_shift
from blocks.layers import down_shifted_conv2d, down_right_shifted_conv2d, down_shift, right_shift
from blocks.losses import bernoulli_loss
from blocks.samplers import gaussian_sampler, mix_logistic_sampler, bernoulli_sampler
from blocks.helpers import int_shape, broadcast_masks_tf


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
        with arg_scope([gated_resnet], nonlinearity=nonlinearity, dropout_p=dropout_p, counters=self.counters):
            with arg_scope([gated_resnet, down_shifted_conv2d, down_right_shifted_conv2d], bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training):
                xs = int_shape(x)
                x_pad = tf.concat([x,tf.ones(xs[:-1]+[1])],3) # add channel of ones to distinguish image from padding later on

                u_list = [down_shift(down_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2, 3]))] # stream for pixels above
                ul_list = [down_shift(down_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[1,3])) + \
                        right_shift(down_right_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2,1]))] # stream for up and to the left
                receptive_field = (2, 3)
                for rep in range(nr_resnet):
                    u_list.append(gated_resnet(u_list[-1], conv=down_shifted_conv2d))
                    ul_list.append(gated_resnet(ul_list[-1], u_list[-1], conv=down_right_shifted_conv2d))
                    receptive_field = (receptive_field[0]+1, receptive_field[1]+2)
                x_out = nin(tf.nn.elu(ul_list[-1]), 1, nonlinearity=None)
                print("    * receptive_field", receptive_field)
                return x_out

    def _loss(self, x, outputs):
        l =  tf.reduce_mean(bernoulli_loss(x, outputs, sum_all=False))
        return l
