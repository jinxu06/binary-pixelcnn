import os
import sys
import json
import argparse
import time
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from args import argument_parser, prepare_args
from blocks.helpers import visualize_samples, get_nonlinearity, int_shape, get_trainable_variables
from blocks.optimizers import multi_gpu_adam_optimizer
import data.mnist as mnist
from models.binary_pixelcnn import BinaryPixelCNN
from blocks.plots import visualize_samples
#from data.omniglot import OmniglotDataSource, Omniglot
import data.omniglot as omniglot
from data.dataset import Dataset
from learners.fomaml import FOMAML

parser = argument_parser()
args = parser.parse_args()
args = prepare_args(args)

meta_train_set, meta_eval_set = omniglot.load("/data/ziz/not-backed-up/jxu/omniglot", args.batch_size, num_train=1200, augment_train_set=True, one_hot=True)


models = [BinaryPixelCNN(counters={}) for i in range(args.nr_model)]
model_opt = {
    "img_size": args.img_size,
    "batch_size": args.batch_size,
    "nr_model": args.nr_model,
    "nr_resnet": 3,
    "nr_filters": 30,
    "nonlinearity": tf.nn.elu,
    "bn": False,
    "kernel_initializer": tf.contrib.layers.xavier_initializer(),
    "kernel_regularizer":None,
}

model = tf.make_template('model', BinaryPixelCNN.construct)

for i in range(args.nr_model):
    with tf.device('/gpu:%d' % (i%args.nr_gpu)):
        model(models[i], **model_opt)

optimize_op = multi_gpu_adam_optimizer(models, args.nr_gpu, args.learning_rate, params=tf.trainable_variables())

initializer = tf.global_variables_initializer()
saver = tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    if args.debug:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    sess.run(initializer)

    if args.load_params:
        ckpt_file = args.save_dir + '/params_' + args.data_set + '.ckpt'
        print('restoring parameters from', ckpt_file)
        saver.restore(sess, ckpt_file)

    params = {
        "eval_num_tasks": 20,
        "meta_iter_per_epoch": 100,
        "meta_batch_size": 5,
        "meta_step_size": 1e-4,
        "num_shots": 10,
        "test_shots": 10,
        "inner_iter": 5,
        "inner_batch_size": args.batch_size,
    }

    mlearner = FOMAML(session=sess, parallel_models=models, optimize_op=optimize_op, train_set=meta_train_set, eval_set=meta_eval_set, variables=tf.trainable_variables())
    mlearner.run(100, 1, 10, **params)


    # mlearner.run(num_epoch=args.max_num_epoch, eval_interval=1, save_interval=args.save_interval)
