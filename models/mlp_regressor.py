import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope, add_arg_scope
from blocks.layers import conv2d, deconv2d, dense, nin, gated_resnet
from blocks.layers import up_shifted_conv2d, up_left_shifted_conv2d, up_shift, left_shift
from blocks.layers import down_shifted_conv2d, down_right_shifted_conv2d, down_shift, right_shift, down_shifted_deconv2d, down_right_shifted_deconv2d
from blocks.losses import bernoulli_loss
from blocks.samplers import gaussian_sampler, mix_logistic_sampler, bernoulli_sampler
from blocks.helpers import int_shape, broadcast_masks_tf, get_name, get_trainable_variables

class MLPRegressor(object):

    def __init__(self, counters={}):
        self.counters = counters

    def construct(self, mlp, obs_shape, alpha=0.01, nonlinearity=tf.nn.relu, bn=False, kernel_initializer=None, kernel_regularizer=None):
        #
        self.mlp = mlp
        self.obs_shape = obs_shape
        self.alpha = alpha
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

        # self.use_z_ph = tf.cast(tf.placeholder_with_default(False, shape=()), dtype=tf.float32)
        # self.z_ph = tf.placeholder_with_default(np.zeros((1, self.z_dim), dtype=np.float32), shape=(1, self.z_dim))
        #
        self.y_hat = self._model()
        self.preds = self.y_hat
        self.loss = self._loss()
        #
        self.grads = tf.gradients(self.loss, tf.trainable_variables(), colocate_gradients_with_ops=True)

    def _model(self):
        default_args = {
            "nonlinearity": self.nonlinearity,
            "bn": self.bn,
            "kernel_initializer": self.kernel_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "is_training": self.is_training,
            "counters": self.counters,
        }
        with arg_scope([self.mlp], **default_args):
            y_hat = self.mlp(self.X_c, scope='mlp')
            vars = get_trainable_variables(['mlp'])
            for k in range(1):
                loss = tf.losses.mean_squared_error(labels=self.y_c, predictions=y_hat)
                grads = tf.gradients(loss, vars, colocate_gradients_with_ops=True)
                vars = [v - self.alpha * g for v, g in zip(vars, grads)]
                y_hat = self.mlp(self.X_c, scope='mlp-{0}'.format(k), params=vars.copy())
            y_hat_test = self.mlp(self.X_t, scope='mlp-test', params=vars.copy())
            return y_hat_test

    def _loss(self):
        return tf.losses.mean_squared_error(labels=self.y_t, predictions=self.preds)


    def predict(self, sess, X_c_value, y_c_value, X_t_value):
        feed_dict = {
            self.X_c: X_c_value,
            self.y_c: y_c_value,
            self.X_t: X_t_value,
            self.y_t: np.zeros((X_t_value.shape[0],)),
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



from blocks.layers_beta import dense
@add_arg_scope
def mlp(X, scope="mlp", params=None, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, is_training=False, counters={}):
    name = get_name(scope, counters)
    print("construct", name, "...")
    if params is not None:
        params.reverse()
    with tf.variable_scope(name):
        default_args = {
            "nonlinearity": nonlinearity,
            "bn": bn,
            "kernel_initializer": kernel_initializer,
            "kernel_regularizer": kernel_regularizer,
            "is_training": is_training,
            "counters": counters,
        }
        with arg_scope([dense], **default_args):
            batch_size = tf.shape(X)[0]
            size = 256
            outputs = X
            for k in range(4):
                if params is not None:
                    outputs = dense(outputs, size, W=params.pop(), b=params.pop())
                else:
                    outputs = dense(outputs, size)
            if params is not None:
                outputs = dense(outputs, 1, nonlinearity=None, W=params.pop(), b=params.pop())
            else:
                outputs = dense(outputs, 1, nonlinearity=None)
            outputs = tf.reshape(outputs, shape=(batch_size,))
            return outputs
