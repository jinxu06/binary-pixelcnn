import os
import sys
import json
import argparse
import time
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from blocks.helpers import visualize_samples, get_nonlinearity, int_shape, get_trainable_variables
from blocks.optimizers import multi_gpu_adam_optimizer
import data.mnist as mnist
from models.binary_pixelcnn import BinaryPixelCNN
from blocks.plots import visualize_samples
#from data.omniglot import OmniglotDataSource, Omniglot
import data.omniglot as omniglot
from data.dataset import Dataset
from learners.fomaml import FOMAML


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--debug', help='', action='store_true', default=False)
parser.add_argument('-mne', '--max_num_epoch', type=int, default=200, help="maximal number of epoches")
parser.add_argument('-ds', '--data_set', type=str, default="omniglot", help='dataset name')
parser.add_argument('-is', '--img_size', type=int, default=28, help="size of input image")
parser.add_argument('-bs', '--batch_size', type=int, default=100, help='Batch size during training per GPU')
parser.add_argument('-ng', '--nr_gpu', type=int, default=1, help='How many GPUs to distribute the training across?')
parser.add_argument('-nm', '--nr_model', type=int, default=1, help='How many models are there with shared parameters?')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='Base learning rate')
parser.add_argument('-sd', '--save_dir', type=str, default="/data/ziz/jxu/hmaml-saved-models/test", help='Location for parameter checkpoints and samples')
parser.add_argument('-g', '--gpus', type=str, default="", help='GPU No.s')
parser.add_argument('-s', '--seed', type=int, default=1, help='Random seed to use')
parser.add_argument('-si', '--save_interval', type=int, default=10, help='Every how many epochs to write checkpoint/samples?')
parser.add_argument('-lp', '--load_params', dest='load_params', action='store_true', help='Restore training from previous model checkpoint?')
args = parser.parse_args()

args.nr_gpu = len(args.gpus.split(","))
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
print('input args:\n', json.dumps(vars(args), indent=4, separators=(',',':'))) # pretty print args
if not os.path.exists(args.save_dir) and args.save_dir!="":
    os.makedirs(args.save_dir)

tf.set_random_seed(args.seed)
np.random.seed(args.seed)

meta_train_set, meta_eval_set = omniglot.load("/data/ziz/not-backed-up/jxu/omniglot", args.batch_size, num_train=1200, augment_train_set=True, one_hot=True)

xs = [tf.placeholder(tf.float32, shape=(args.batch_size, args.img_size, args.img_size, 1)) for i in range(args.nr_model)]
is_trainings = [tf.placeholder(tf.bool, shape=()) for i in range(args.nr_model)]
dropout_ps = [tf.placeholder(tf.float32, shape=()) for i in range(args.nr_model)]

models = [BinaryPixelCNN(counters={}) for i in range(args.nr_model)]
model_opt = {
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
        model(models[i], xs[i], is_trainings[i], dropout_ps[i], **model_opt)

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
