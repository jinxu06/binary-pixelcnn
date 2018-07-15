import os
import sys
import json
import argparse
import time
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from blocks.optimizers import adam_updates
from models.mlp_regressor import MLPRegressor
from data.dataset import Dataset
from models.neural_processes import NeuralProcess
from learners.regression_meta_learner import RegressionMetaLearner
from data.sinusoid import Sinusoid, SineWave
from data.dataset import Dataset
import matplotlib.pyplot as plt
plt.style.use("ggplot")

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--debug', help='', action='store_true', default=False)
parser.add_argument('-mne', '--max_num_epoch', type=int, default=200, help="maximal number of epoches")
parser.add_argument('-ds', '--data_set', type=str, default="sinusoid", help='dataset name')
parser.add_argument('-bs', '--batch_size', type=int, default=100, help='Batch size during training per GPU')
parser.add_argument('-zd', '--z_dim', type=int, default=10, help='dimension of z')
parser.add_argument('-ng', '--nr_gpu', type=int, default=1, help='How many GPUs to distribute the training across?')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='Base learning rate')
parser.add_argument('-sd', '--save_dir', type=str, default="", help='Location for parameter checkpoints and samples')
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

# Data IO
train_set = Sinusoid(amp_range=[0.1, 2.0], phase_range=[np.pi, np.pi], period_range=[2*np.pi, 2*np.pi], input_range=[-10,10])
val_set = train_set
# sines = sinusoid.sample(1)
# samples = sines[0].sample(200)
# train_set, val_set = samples[:100], samples[100:]
# train_set = Dataset(batch_size=args.batch_size, X=train_set[:, 0:1], y=train_set[:, 1])
# val_set = Dataset(batch_size=args.batch_size, X=val_set[:, 0:1], y=val_set[:, 1])


# xs = [tf.placeholder(tf.float32, shape=(args.batch_size, 1)) for i in range(args.nr_gpu)]
# ys = [tf.placeholder(tf.float32, shape=(args.batch_size)) for i in range(args.nr_gpu)]
# is_trainings = [tf.placeholder(tf.bool, shape=()) for i in range(args.nr_gpu)]

models = [NeuralProcess(counters={}) for i in range(args.nr_gpu)]
model_opt = {
    "batch_size": args.batch_size,
    "z_dim": args.z_dim,
    "nonlinearity": tf.nn.elu,
    "bn": True,
    "kernel_initializer": tf.contrib.layers.xavier_initializer(),
    "kernel_regularizer":None,
}

model = tf.make_template('model', NeuralProcess.construct)

for i in range(args.nr_gpu):
    with tf.device('/cpu:%d' % i):
        model(models[i], **model_opt)

if True:
    all_params = tf.trainable_variables() #get_trainable_variables(["conv_encoder", "conv_decoder", "conv_pixel_cnn"])

    grads = []
    for i in range(args.nr_gpu):
        with tf.device('/cpu:%d' % i):
            grads.append(tf.gradients(models[i].loss, all_params, colocate_gradients_with_ops=True))
    with tf.device('/cpu:0'):
        for i in range(1, args.nr_gpu):
            for j in range(len(grads[0])):
                grads[0][j] += grads[i][j]

        train_step = adam_updates(all_params, grads[0], lr=args.learning_rate)


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

    learner = RegressionMetaLearner(session=sess, parallel_models=models, optimize_op=train_step, train_set=train_set, eval_set=val_set, variables=all_params)
    learner.run(num_epoch=args.max_num_epoch, eval_interval=1, save_interval=args.save_interval)
