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
from blocks.components import sinusoid_fc_encoder, aggregator
from blocks.estimators import compute_gaussian_kld

class NeuralProcess(object):

    def __init__(self, counters={}):
        self.counters = counters

    def construct(self, batch_size, z_dim, nonlinearity=tf.nn.relu, bn=False, kernel_initializer=None, kernel_regularizer=None):
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.X = tf.placeholder(tf.float32, shape=(self.batch_size, 1))
        self.y = tf.placeholder(tf.float32, shape=(self.batch_size))
        self.nonlinearity = nonlinearity
        self.bn = bn
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.use_z_ph = tf.cast(tf.placeholder_with_default(False, shape=()), dtype=tf.float32)
        self.z_ph = tf.placeholder_with_default(np.zeros((1, self.z_dim), dtype=np.float32), shape=(1, self.z_dim))

        self.outputs = self._model(self.X, self.y, self.nonlinearity, self.bn, self.kernel_initializer, self.kernel_regularizer, self.is_training)
        self.predictions = self.outputs
        self.loss = self._loss(self.y, self.predictions)


    def _model(self, X, y, nonlinearity, bn, kernel_initializer, kernel_regularizer, is_training):
        X_y = tf.concat([X, y[:, None]], axis=1)
        bsize = int_shape(X)
        with arg_scope([dense, sinusoid_fc_encoder], nonlinearity=nonlinearity, bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training, counters=self.counters):
            r = sinusoid_fc_encoder(X_y, 2)
            self.z_mu, self.z_log_sigma_sq = aggregator(r, z_dim=self.z_dim, method=tf.reduce_max)
            self.z_sigma = tf.exp(self.z_log_sigma_sq / 2.)
            z = gaussian_sampler(self.z_mu, self.z_sigma)
            z = tf.reduce_max(X_y[:, 1:], axis=1, keepdims=True)##
            z = (1-self.use_z_ph) * z + self.use_z_ph * self.z_ph
            z = dense(z, 1)
            outputs = dense(X, 100) + z
            outputs = dense(outputs, 100) + z
            outputs = dense(outputs, 100) + z
            outputs = dense(outputs, 1, nonlinearity=None)
            outputs = tf.reshape(outputs, shape=(bsize[0],))
            return outputs


    def _loss(self, y, predictions):
        self.beta = 1.
        self.reg = compute_gaussian_kld(self.z_mu, self.z_log_sigma_sq)
        self.nll = tf.losses.mean_squared_error(labels=y, predictions=predictions)
        return self.nll + self.beta * self.reg
