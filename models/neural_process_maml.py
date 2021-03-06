import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope, add_arg_scope
from blocks.layers import conv2d, deconv2d, dense, nin, gated_resnet
from blocks.layers import up_shifted_conv2d, up_left_shifted_conv2d, up_shift, left_shift
from blocks.layers import down_shifted_conv2d, down_right_shifted_conv2d, down_shift, right_shift, down_shifted_deconv2d, down_right_shifted_deconv2d
from blocks.losses import sum_squared_error
from blocks.samplers import gaussian_sampler
from blocks.helpers import int_shape, broadcast_masks_tf, get_name, get_trainable_variables
from blocks.estimators import compute_2gaussian_kld



class NeuralProcessMAML(object):

    def __init__(self, counters={}, user_mode='train'):
        self.counters = counters
        self.user_mode = user_mode

    def construct(self, sample_encoder, aggregator, conditional_decoder, obs_shape, r_dim, z_dim, alpha=0.01, nonlinearity=tf.nn.relu, bn=False, kernel_initializer=None, kernel_regularizer=None):
        #
        self.sample_encoder = sample_encoder
        self.aggregator = aggregator
        self.conditional_decoder = conditional_decoder
        self.obs_shape = obs_shape
        self.r_dim = r_dim
        self.z_dim = z_dim
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
        self.use_z_ph = tf.cast(tf.placeholder_with_default(False, shape=()), dtype=tf.float32)
        self.z_ph = tf.placeholder_with_default(np.zeros((1, self.z_dim), dtype=np.float32), shape=(1, self.z_dim))
        #
        self.y_hat = self._model()
        self.preds = self.y_hat
        self.loss = self._loss(beta=1.0, y_sigma=0.2)
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
        with arg_scope([self.conditional_decoder], **default_args):
            default_args.update({"bn":False})
            with arg_scope([self.sample_encoder, self.aggregator], **default_args):
                num_c = tf.shape(self.X_c)[0]
                X_ct = tf.concat([self.X_c, self.X_t], axis=0)
                y_ct = tf.concat([self.y_c, self.y_t], axis=0)
                r_ct = self.sample_encoder(X_ct, y_ct, self.r_dim)

                self.z_mu_pr, self.z_log_sigma_sq_pr, self.z_mu_pos, self.z_log_sigma_sq_pos = self.aggregator(r_ct, num_c, self.z_dim)
                if self.user_mode == 'train':
                    z = gaussian_sampler(self.z_mu_pos, tf.exp(0.5*self.z_log_sigma_sq_pos))
                elif self.user_mode == 'eval':
                    z = self.z_mu_pos
                else:
                    raise Exception("unknown user_mode")
                z = (1-self.use_z_ph) * z + self.use_z_ph * self.z_ph

                # add maml ops
                y_hat = self.conditional_decoder(self.X_c, z)
                vars = get_trainable_variables(['conditional_decoder'])
                inner_iters = 1
                eval_iters = 10
                y_hat_test_arr = [self.conditional_decoder(self.X_t, z, params=vars.copy())]
                for k in range(1, max(inner_iters, eval_iters)+1):
                    loss = sum_squared_error(labels=self.y_c, predictions=y_hat)
                    grads = tf.gradients(loss, vars, colocate_gradients_with_ops=True)
                    vars = [v - self.alpha * g for v, g in zip(vars, grads)]
                    y_hat = self.conditional_decoder(self.X_c, z, params=vars.copy())
                    y_hat_test = self.conditional_decoder(self.X_t, z, params=vars.copy())
                    y_hat_test_arr.append(y_hat_test)
                self.eval_ops = y_hat_test_arr
                return y_hat_test_arr[inner_iters]



    def _loss(self, beta=1., y_sigma=1./np.sqrt(2)):
        self.reg = compute_2gaussian_kld(self.z_mu_pr, self.z_log_sigma_sq_pr, self.z_mu_pos, self.z_log_sigma_sq_pos)
        self.nll = sum_squared_error(labels=self.y_t, predictions=self.y_hat)
        return self.nll / (2*y_sigma**2) + beta * self.reg

    def predict(self, sess, X_c_value, y_c_value, X_t_value, step=None):
        feed_dict = {
            self.X_c: X_c_value,
            self.y_c: y_c_value,
            self.X_t: X_t_value,
            self.y_t: np.zeros((X_t_value.shape[0],)),
            self.is_training: False,
        }
        z_mu, z_log_sigma_sq = sess.run([self.z_mu_pr, self.z_log_sigma_sq_pr], feed_dict=feed_dict)
        z_sigma = np.exp(0.5*z_log_sigma_sq)
        z_pr = np.random.normal(loc=z_mu, scale=z_sigma)
        feed_dict.update({
            self.use_z_ph: True,
            self.z_ph: z_pr,
        })
        if step is None:
            preds= sess.run(self.preds, feed_dict=feed_dict)
        else:
            preds= sess.run(self.eval_ops[step], feed_dict=feed_dict)
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


from blocks.layers_beta import dense

@add_arg_scope
def fc_encoder(X, y, r_dim, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, is_training=False, counters={}):
    inputs = tf.concat([X, y[:, None]], axis=1)
    name = get_name("fc_encoder", counters)
    print("construct", name, "...")
    with tf.variable_scope(name):
        with arg_scope([dense], nonlinearity=nonlinearity, bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training, counters=counters):
            size = 256
            outputs = dense(inputs, size)
            outputs = nonlinearity(dense(outputs, size, nonlinearity=None) + dense(inputs, size, nonlinearity=None))
            inputs = outputs
            outputs = dense(outputs, size)
            outputs = nonlinearity(dense(outputs, size, nonlinearity=None) + dense(inputs, size, nonlinearity=None))
            outputs = dense(outputs, size)
            outputs = dense(outputs, r_dim, nonlinearity=None, bn=False)
            return outputs

@add_arg_scope
def aggregator(r, num_c, z_dim, method=tf.reduce_mean, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, is_training=False, counters={}):
    name = get_name("aggregator", counters)
    print("construct", name, "...")
    with tf.variable_scope(name):
        with arg_scope([dense], nonlinearity=nonlinearity, bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training, counters=counters):
            r_pr = method(r[:num_c], axis=0, keepdims=True)
            r = method(r, axis=0, keepdims=True)
            r = tf.concat([r_pr, r], axis=0)
            size = 256
            r = dense(r, size)
            r = dense(r, size)
            r = dense(r, size)
            z_mu = dense(r, z_dim, nonlinearity=None, bn=False)
            z_log_sigma_sq = dense(r, z_dim, nonlinearity=None, bn=False)
            return z_mu[:1], z_log_sigma_sq[:1], z_mu[1:], z_log_sigma_sq[1:]

@add_arg_scope
def conditional_decoder(x, z, params=None, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, is_training=False, counters={}):
    name = get_name("conditional_decoder", counters)
    print("construct", name, "...")
    if params is not None:
        params.reverse()
    with tf.variable_scope(name):
        with arg_scope([dense], nonlinearity=nonlinearity, bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training, counters=counters):
            size = 256
            batch_size = tf.shape(x)[0]
            x = tf.tile(x, tf.stack([1, int_shape(z)[1]]))
            z = tf.tile(z, tf.stack([batch_size, 1]))
            # xz = x + z * tf.get_variable(name="coeff", shape=(), dtype=tf.float32, initializer=tf.constant_initializer(2.0))
            xz = x

            if params is not None:
                a = dense(xz, size, nonlinearity=None, W=params.pop(), b=params.pop()) + dense(z, size, nonlinearity=None, W=params.pop(), b=params.pop())
                outputs = tf.nn.tanh(a) * tf.sigmoid(a)

                for k in range(4):
                    a = dense(outputs, size, nonlinearity=None, W=params.pop(), b=params.pop()) + dense(z, size, nonlinearity=None, W=params.pop(), b=params.pop())
                    outputs = tf.nn.tanh(a) * tf.sigmoid(a)
                outputs = dense(outputs, 1, nonlinearity=None, bn=False, W=params.pop(), b=params.pop())
                outputs = tf.reshape(outputs, shape=(batch_size,))
                return outputs
            else:
                a = dense(xz, size, nonlinearity=None) + dense(z, size, nonlinearity=None)
                outputs = tf.nn.tanh(a) * tf.sigmoid(a)

                for k in range(4):
                    a = dense(outputs, size, nonlinearity=None) + dense(z, size, nonlinearity=None)
                    outputs = tf.nn.tanh(a) * tf.sigmoid(a)
                outputs = dense(outputs, 1, nonlinearity=None, bn=False)
                outputs = tf.reshape(outputs, shape=(batch_size,))
                return outputs
