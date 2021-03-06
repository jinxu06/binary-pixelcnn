import matplotlib
matplotlib.use('Agg')
import os
import sys
import json
import argparse
import time
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from args import argument_parser, prepare_args, model_kwards, learn_kwards
from blocks.helpers import visualize_samples, get_nonlinearity, int_shape, get_trainable_variables
from blocks.optimizers import multi_gpu_adam_optimizer
from data.dataset import Dataset
from data.gpsampler import GPSampler

from models.mlp_regressor import MLPRegressor, mlp


parser = argument_parser()
args = parser.parse_args()
args = prepare_args(args)

data = np.load("gpsamples_var05.npz")
train_data = {"xs":data['xs'][:50000], "ys":data['ys'][:50000]}
val_data = {"xs":data['xs'][50000:60000], "ys":data['ys'][50000:60000]}
train_set = GPSampler(input_range=[-2., 2.], var_range=[0.5, 0.5], max_num_samples=200, data=train_data)
val_set = GPSampler(input_range=[-2., 2.], var_range=[0.5, 0.5], max_num_samples=200, data=val_data)

model_opt = {
    "mlp": mlp,
    "obs_shape": [1],
    "nonlinearity": tf.nn.relu,
    "bn": False,
    "kernel_initializer": tf.contrib.layers.xavier_initializer(uniform=False),
    "kernel_regularizer":None,
}
model = MLPRegressor()
model.construct(**model_opt)
print(model)


# models = [NeuralProcess(counters={}) for i in range(args.nr_model)]
#
# from blocks.components import fc_encoder, aggregator, conditional_decoder
#
# model_opt = {
#     "sample_encoder": fc_encoder,
#     "aggregator": aggregator,
#     "conditional_decoder": conditional_decoder,
#     "obs_shape": [1],
#     "r_dim": 128,
#     "z_dim": 32,
#     "nonlinearity": tf.nn.relu,
#     "bn": False,
#     "kernel_initializer": tf.contrib.layers.xavier_initializer(uniform=False),
#     "kernel_regularizer":None,
# }
#
# model = tf.make_template('model', NeuralProcess.construct)
#
# for i in range(args.nr_model):
#     with tf.device('/'+ args.device_type +':%d' % (i%args.nr_gpu)):
#         model(models[i], **model_opt)
#
# save_dir = "/data/ziz/jxu/neural_processes/test-sum-ysigma"
# learner = NPLearner(session=None, parallel_models=models, optimize_op=None, train_set=train_set, eval_set=val_set, variables=tf.trainable_variables(), lr=args.learning_rate, device_type=args.device_type, save_dir=save_dir)
#
#
# initializer = tf.global_variables_initializer()
# saver = tf.train.Saver()
#
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# with tf.Session(config=config) as sess:
#     if args.debug:
#         sess = tf_debug.LocalCLIDebugWrapperSession(sess)
#     sess.run(initializer)
#
#     learner.set_session(sess)
#
#     # summary_writer = tf.summary.FileWriter('logdir', sess.graph)
#
#     run_params = {
#         "num_epoch": 500,
#         "eval_interval": 5,
#         "save_interval": args.save_interval,
#         "eval_samples": 1000,
#         "meta_batch": args.nr_model,
#         "num_shots": 10,
#         "test_shots": 10,
#         "load_params": args.load_params,
#     }
#     learner.run(**run_params)
