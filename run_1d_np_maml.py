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

parser = argument_parser()
args = parser.parse_args()
args = prepare_args(args)

if args.dataset_name == 'gpsamples':
    from data.gpsampler import GPSampler
    data = np.load("gpsamples_var05.npz")
    train_data = {"xs":data['xs'][:50000], "ys":data['ys'][:50000]}
    val_data = {"xs":data['xs'][50000:60000], "ys":data['ys'][50000:60000]}
    train_set = GPSampler(input_range=[-2., 2.], var_range=[0.5, 0.5], max_num_samples=200, data=train_data)
    val_set = GPSampler(input_range=[-2., 2.], var_range=[0.5, 0.5], max_num_samples=200, data=val_data)
elif args.dataset_name == 'sinusoid':
    from data.sinusoid import Sinusoid
    train_set = Sinusoid(amp_range=[0.1, 5.0], phase_range=[0, np.pi], period_range=[2*np.pi, 2*np.pi], input_range=[-5., 5.], dataset_name=args.dataset_name)
    val_set = train_set
elif args.dataset_name == 'sinusoid-var-period':
    from data.sinusoid import Sinusoid
    train_set = Sinusoid(amp_range=[0.1, 5.0], phase_range=[0, np.pi], period_range=[1*np.pi, 4*np.pi], input_range=[-5., 5.], dataset_name=args.dataset_name)
    val_set = train_set
elif args.dataset_name == 'distorted-sinusoid':
    from data.distorted_sinusoid import DistortedSinusoid
    from data.gpsampler import GPSampler
    data = np.load("gpsamples_var05.npz")
    train_data = {"xs":data['xs'][:50000], "ys":data['ys'][:50000]}
    val_data = {"xs":data['xs'][50000:60000], "ys":data['ys'][50000:60000]}
    train_set = GPSampler(input_range=[-2., 2.], var_range=[0.5, 0.5], max_num_samples=200, data=train_data)
    val_set = GPSampler(input_range=[-2., 2.], var_range=[0.5, 0.5], max_num_samples=200, data=val_data)
    from data.sinusoid import Sinusoid
    sinusoid_set = Sinusoid(amp_range=[1., 5.0], phase_range=[0, np.pi], period_range=[0.4*np.pi, 0.8*np.pi], input_range=[-2., 2.])
    train_set = DistortedSinusoid(sinusoid_set, train_set, noise_level=0.5, dataset_name=args.dataset_name)
    val_set = DistortedSinusoid(sinusoid_set, val_set, noise_level=0.5, dataset_name=args.dataset_name)
else:
    raise Exception("Dataset {0} not found".format(args.dataset_name))

from models.neural_process_maml import NeuralProcessMAML
from learners.neural_process_maml_learner import NeuralProcessMAMLLearner

models = [NeuralProcessMAML(counters={}) for i in range(args.nr_model)]

from blocks.components import fc_encoder, aggregator
from models.neural_process_maml import conditional_decoder

model_opt = {
    "sample_encoder": fc_encoder,
    "aggregator": aggregator,
    "conditional_decoder": conditional_decoder,
    "obs_shape": [1],
    "r_dim": 128,
    "z_dim": 32,
    "nonlinearity": tf.nn.relu,
    "bn": False,
    "kernel_initializer": tf.contrib.layers.xavier_initializer(uniform=False),
    "kernel_regularizer":None,
}

model = tf.make_template('model', NeuralProcessMAML.construct)

for i in range(args.nr_model):
    with tf.device('/'+ args.device_type +':%d' % (i%args.nr_gpu)):
        model(models[i], **model_opt)

save_dir = "/data/ziz/jxu/neural_processes_maml/test-{0}".format(args.dataset_name)
learner = NeuralProcessMAMLLearner(session=None, parallel_models=models, optimize_op=None, train_set=train_set, eval_set=val_set, variables=tf.trainable_variables(), lr=args.learning_rate, device_type=args.device_type, save_dir=save_dir)


initializer = tf.global_variables_initializer()
saver = tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    if args.debug:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    sess.run(initializer)

    learner.set_session(sess)

    # summary_writer = tf.summary.FileWriter('logdir', sess.graph)

    run_params = {
        "num_epoch": 500,
        "eval_interval": 5,
        "save_interval": args.save_interval,
        "eval_samples": 1000,
        "meta_batch": args.nr_model,
        "num_shots": 10,
        "test_shots": 10,
        "load_params": args.load_params,
    }
    learner.run(**run_params)
