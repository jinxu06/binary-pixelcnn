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
from blocks.components import fc_encoder, aggregator, conditional_decoder
from blocks.estimators import compute_2gaussian_kld



class NeuralProcess(object):

    def __init__(self, counters={}):
        self.counters = counters

    def construct(self, sample_encoder, aggregator, conditional_decoder, obs_shape, r_dim, z_dim, nonlinearity=tf.nn.relu, bn=False, kernel_initializer=None, kernel_regularizer=None):
        #
        self.sample_encoder = sample_encoder
        self.aggregator = aggregator
        self.conditional_decoder = conditional_decoder
        self.obs_shape = obs_shape
        self.r_dim = r_dim
        self.z_dim = z_dim
        self.nonlinearity = nonlinearity
        self.bn = bn
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        #
        self.X_c = tf.placeholder(tf.float32, shape=tuple([None,]+obs_shape))
        self.y_c = tf.placeholder(tf.float32, shape=(None,))
        self.X_t = tf.placeholder(tf.float32, shape=tuple([None,]+obs_shape))
        self.y_t = tf.placeholder(tf.float32, shape=(None,))
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.use_z_ph = tf.cast(tf.placeholder_with_default(False, shape=()), dtype=tf.float32)
        self.z_ph = tf.placeholder_with_default(np.zeros((1, self.z_dim), dtype=np.float32), shape=(1, self.z_dim))
        #
        self.y_hat = self._model()
        self.preds = self.y_hat
        self.loss = self._loss()
        #
        self.grads = tf.gradients(self.loss, tf.trainable_variables(), colocate_gradients_with_ops=True)

    def _model(self):
        default_args = {
            "nonlinearity": tf.nn.relu,
            "bn": False,
            "kernel_initializer": self.kernel_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "is_training": self.is_training,
            "counters": self.counters,
        }
        with arg_scope([self.sample_encoder, self.aggregator, self.conditional_decoder], **default_args):
            r_c = self.sample_encoder(self.X_c, self.y_c, self.r_dim)
            r_t = self.sample_encoder(self.X_t, self.y_t, self.r_dim)
            r_ct = tf.concat([r_c, r_t], axis=0)
            self.z_mu_pr, self.z_log_sigma_sq_pr = self.aggregator(r_c, self.z_dim)
            self.z_mu_pos, self.z_log_sigma_sq_pos = self.aggregator(r_ct, self.z_dim)
            z = gaussian_sampler(self.z_mu_pr, self.z_log_sigma_sq_pr)
            z = (1-self.use_z_ph) * z + self.use_z_ph * self.z_ph
            y_hat = self.conditional_decoder(self.X_t, z)
            return y_hat

    def _loss(self, beta=1.0):
        self.reg = compute_2gaussian_kld(self.z_mu_pr, self.z_log_sigma_sq_pr, self.z_mu_pos, self.z_log_sigma_sq_pos)
        self.nll = tf.losses.mean_squared_error(labels=self.y_t, predictions=self.y_hat)
        return self.nll + beta * self.reg

    def predict(self, sess, X_c_value, y_c_value, X_t_value):
        feed_dict = {
            self.X_c: X_c_value,
            self.y_c: y_c_value,
            self.X_t: X_t_value,
            self.is_training: False,
        }
        preds= sess.run(self.preds, feed_dict=feed_dict)
        return preds

    def manipulate_z(self, sess, z_value, X_t_value):
        feed_dict = {
            self.use_z_ph: True,
            self.z_ph: z_value,
            self.X_t: X_t_value,
            self.is_training: False,
        }
        preds= sess.run(self.preds, feed_dict=feed_dict)
        return preds

    def compute_loss(self, sess, X_c_value, y_c_value, X_t_value, y_t_value, is_training):
        feed_dict = {
            self.X_c: X_c_value,
            self.y_c: y_c_value,
            self.X_t: X_t_value,
            self.y_t: y_t_value,
            self.is_training: is_training,
        }
        l = sess.run(self.loss, feed_dict=feed_dict)
        return l






# class NeuralProcess(object):
#
#     def __init__(self, counters={}):
#         self.counters = counters
#
#     def construct(self, batch_size, z_dim, nonlinearity=tf.nn.relu, bn=False, kernel_initializer=None, kernel_regularizer=None):
#         self.batch_size = batch_size
#         self.z_dim = z_dim
#         self.X_c = tf.placeholder(tf.float32, shape=(self.batch_size, 1))
#         self.y_c = tf.placeholder(tf.float32, shape=(self.batch_size))
#         self.X_t = tf.placeholder(tf.float32, shape=(self.batch_size, 1))
#         self.y_t = tf.placeholder(tf.float32, shape=(self.batch_size))
#         self.nonlinearity = nonlinearity
#         self.bn = bn
#         self.kernel_initializer = kernel_initializer
#         self.kernel_regularizer = kernel_regularizer
#         self.is_training = tf.placeholder(tf.bool, shape=())
#         self.use_z_ph = tf.cast(tf.placeholder_with_default(False, shape=()), dtype=tf.float32)
#         self.z_ph = tf.placeholder_with_default(np.zeros((1, self.z_dim), dtype=np.float32), shape=(1, self.z_dim))
#
#         self.outputs = self._model(self.X_c, self.y_c, self.X_t, self.nonlinearity, self.bn, self.kernel_initializer, self.kernel_regularizer, self.is_training)
#         self.predictions = self.outputs
#         self.loss = self._loss(self.y_t, self.predictions)
#
#
#     def _model(self, X_c, y_c, X_t, nonlinearity, bn, kernel_initializer, kernel_regularizer, is_training):
#         X_y = tf.concat([X_c, y_c[:, None]], axis=1)
#         with arg_scope([fc_encoder, conditional_decoder], nonlinearity=nonlinearity, bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training, counters=self.counters):
#             r = fc_encoder(X_y, r_dim=5)
#             self.z_mu, self.z_log_sigma_sq = aggregator(r, z_dim=self.z_dim, method=tf.reduce_max)
#             self.z_sigma = tf.exp(self.z_log_sigma_sq / 2.)
#             z = gaussian_sampler(self.z_mu, self.z_sigma)
#             z = (1-self.use_z_ph) * z + self.use_z_ph * self.z_ph
#             y_hat = conditional_decoder(X_t, z)
#             return y_hat
#
#
#     def _loss(self, y_t, predictions):
#         self.beta = 1.
#         self.reg = compute_gaussian_kld(self.z_mu, self.z_log_sigma_sq)
#         self.nll = tf.losses.mean_squared_error(labels=y_t, predictions=predictions)
#         return self.nll + self.beta * self.reg
