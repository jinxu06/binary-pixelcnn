import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope, add_arg_scope
from blocks.layers import conv2d, deconv2d, dense
from blocks.samplers import gaussian_sampler, mix_logistic_sampler
from blocks.losses import mix_logistic_loss
from blocks.helpers import int_shape, broadcast_masks_tf


class BinaryPixelCNN(object):

    def __init__(self, counters={}):
        self.counters = counters

    def __model(self):
        pass

    def __loss(self):
        pass



@add_arg_scope
def cond_pixel_cnn(x, gh=None, sh=None, nonlinearity=tf.nn.elu, nr_resnet=5, nr_filters=100, nr_logistic_mix=10, bn=False, dropout_p=0.0, kernel_initializer=None, kernel_regularizer=None, is_training=False, counters={}):
    name = get_name("conv_pixel_cnn", counters)
    print("construct", name, "...")
    print("    * nr_resnet: ", nr_resnet)
    print("    * nr_filters: ", nr_filters)
    print("    * nr_logistic_mix: ", nr_logistic_mix)
    assert not bn, "auto-reggressive model should not use batch normalization"
    with tf.variable_scope(name):
        with arg_scope([gated_resnet], gh=gh, sh=sh, nonlinearity=nonlinearity, dropout_p=dropout_p, counters=counters):
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
                x_out = nin(tf.nn.elu(ul_list[-1]), 10*nr_logistic_mix)
                print("    * receptive_field", receptive_field)
                return x_out
