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


class BinaryPixelCNN(object):

    def __init__(self, counters={}):
        self.counters = counters

    def construct(self, device, img_size, batch_size, nr_resnet=1, nr_filters=50, nonlinearity=tf.nn.relu, bn=False, kernel_initializer=None, kernel_regularizer=None):
        self.device = device
        self.img_size = img_size
        self.batch_size = batch_size
        self.nr_resnet = nr_resnet
        self.nr_filters = nr_filters
        self.nonlinearity = nonlinearity
        self.bn = bn

        self.X = tf.placeholder(tf.float32, shape=(batch_size, img_size, img_size, 1))
        self.dropout_p = tf.placeholder(tf.float32, shape=())

        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.is_training = tf.placeholder(tf.bool, shape=())

        self.outputs = self._model(self.X, self.nr_resnet, self.nr_filters, self.nonlinearity, self.dropout_p, self.bn, self.kernel_initializer, self.kernel_regularizer, self.is_training)
        self.loss = self._loss(self.X, self.outputs)

        self.x_hat = bernoulli_sampler(self.outputs)


    def construct_maml_ops(self, params, inner_step_size, inner_iters):
        self.params = params
        gradients =  tf.gradients(self.loss, self.params, colocate_gradients_with_ops=True)
        print(gradients)
        self.fast_params = [p - self.inner_step_size * grad for p, grad in zip(self.params, gradients)]
        print(self.fast_params)
        for i in range(inner_iters-1):
            gradients =  tf.gradients(self.loss, self.fast_params, colocate_gradients_with_ops=True)
            self.fast_params = [p - self.inner_step_size * grad for p, grad in zip(fast_params, gradients)]

    def _model(self, x, nr_resnet, nr_filters, nonlinearity, dropout_p, bn, kernel_initializer, kernel_regularizer, is_training):
        with arg_scope([gated_resnet], nonlinearity=nonlinearity, dropout_p=dropout_p, counters=self.counters):
            with arg_scope([gated_resnet, down_shifted_conv2d, down_right_shifted_conv2d, down_shifted_deconv2d, down_right_shifted_deconv2d], bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training, counters=self.counters):

                # ////////// up pass through pixelCNN ////////
                xs = int_shape(x)
                #ap = tf.Variable(np.zeros((xs[1], xs[2], 1), dtype=np.float32), trainable=True)
                #aps = tf.stack([ap for _ in range(xs[0])], axis=0)
                x_pad = tf.concat([x, tf.ones(xs[:-1] + [1])], 3)

                u_list = [down_shift(down_shifted_conv2d(
                    x_pad, num_filters=nr_filters, filter_size=[2, 3]))]  # stream for pixels above
                ul_list = [down_shift(down_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[1, 3])) +
                           right_shift(down_right_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2, 1]))]  # stream for up and to the left

                for rep in range(nr_resnet):
                    u_list.append(gated_resnet(
                        u_list[-1], conv=down_shifted_conv2d))
                    ul_list.append(gated_resnet(
                        ul_list[-1], u_list[-1], conv=down_right_shifted_conv2d))

                u_list.append(down_shifted_conv2d(
                    u_list[-1], num_filters=nr_filters, strides=[2, 2]))
                ul_list.append(down_right_shifted_conv2d(
                    ul_list[-1], num_filters=nr_filters, strides=[2, 2]))

                for rep in range(nr_resnet):
                    u_list.append(gated_resnet(
                        u_list[-1], conv=down_shifted_conv2d))
                    ul_list.append(gated_resnet(
                        ul_list[-1], u_list[-1], conv=down_right_shifted_conv2d))

                u_list.append(down_shifted_conv2d(
                    u_list[-1], num_filters=nr_filters, strides=[2, 2]))
                ul_list.append(down_right_shifted_conv2d(
                    ul_list[-1], num_filters=nr_filters, strides=[2, 2]))

                for rep in range(nr_resnet):
                    u_list.append(gated_resnet(
                        u_list[-1], conv=down_shifted_conv2d))
                    ul_list.append(gated_resnet(
                        ul_list[-1], u_list[-1], conv=down_right_shifted_conv2d))

                # /////// down pass ////////

                u = u_list.pop()
                ul = ul_list.pop()
                for rep in range(nr_resnet):
                    u = gated_resnet(
                        u, u_list.pop(), conv=down_shifted_conv2d)
                    ul = gated_resnet(ul, tf.concat(
                        [u, ul_list.pop()], 3), conv=down_right_shifted_conv2d)

                u = down_shifted_deconv2d(
                    u, num_filters=nr_filters, strides=[2, 2])
                ul = down_right_shifted_deconv2d(
                    ul, num_filters=nr_filters, strides=[2, 2])

                for rep in range(nr_resnet + 1):
                    u = gated_resnet(
                        u, u_list.pop(), conv=down_shifted_conv2d)
                    ul = gated_resnet(ul, tf.concat(
                        [u, ul_list.pop()], 3), conv=down_right_shifted_conv2d)

                u = down_shifted_deconv2d(
                    u, num_filters=nr_filters, strides=[2, 2])
                ul = down_right_shifted_deconv2d(
                    ul, num_filters=nr_filters, strides=[2, 2])

                for rep in range(nr_resnet + 1):
                    u = gated_resnet(
                        u, u_list.pop(), conv=down_shifted_conv2d)
                    ul = gated_resnet(ul, tf.concat(
                        [u, ul_list.pop()], 3), conv=down_right_shifted_conv2d)

                x_out = nin(tf.nn.elu(ul), 1)

                assert len(u_list) == 0
                assert len(ul_list) == 0

                return x_out

    def _loss(self, x, outputs):
        l =  tf.reduce_mean(bernoulli_loss(x, outputs, sum_all=False))
        return l
