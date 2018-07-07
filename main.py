import os
import sys
import json
import argparse
import time
import numpy as np
import tensorflow as tf
from blocks.helpers import visualize_samples, get_nonlinearity, int_shape, get_trainable_variables
from blocks.optimizers import adam_updates
import data.mnist as mnist
from models.binary_pixelcnn import BinaryPixelCNN

parser = argparse.ArgumentParser()
parser.add_argument('-is', '--img_size', type=int, default=28, help="size of input image")
parser.add_argument('-bs', '--batch_size', type=int, default=100, help='Batch size during training per GPU')
parser.add_argument('-ng', '--nr_gpu', type=int, default=1, help='How many GPUs to distribute the training across?')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='Base learning rate')
args = parser.parse_args()

datasets = mnist.load(data_dir="~/scikit_learn_data", num_classes=5, batch_size=args.batch_size, split=[5./7, 1./7, 1./7])
train_set, val_set = datasets[0], datasets[1]

xs = [tf.placeholder(tf.float32, shape=(args.batch_size, args.img_size, args.img_size, 1)) for i in range(args.nr_gpu)]
is_trainings = [tf.placeholder(tf.bool, shape=()) for i in range(args.nr_gpu)]
dropout_ps = [tf.placeholder(tf.float32, shape=()) for i in range(args.nr_gpu)]

models = [BinaryPixelCNN(counters={}) for i in range(args.nr_gpu)]
model_opt = {
    "nr_resnet": 1,
    "nr_filters": 50,
    "nonlinearity": tf.nn.relu,
    "bn": False,
    "kernel_initializer": tf.contrib.layers.xavier_initializer(),
    "kernel_regularizer":None,
}

model = tf.make_template('model', BinaryPixelCNN.construct)

for i in range(args.nr_gpu):
    with tf.device('/gpu:%d' % i):
        model(models[i], xs[i], is_trainings[i], dropout_ps[i], **model_opt)

if True:
    all_params = tf.trainable_variables() #get_trainable_variables(["conv_encoder", "conv_decoder", "conv_pixel_cnn"])
    grads = []
    for i in range(args.nr_gpu):
        with tf.device('/gpu:%d' % i):
            grads.append(tf.gradients(models[i].loss, all_params, colocate_gradients_with_ops=True))
    with tf.device('/gpu:0'):
        for i in range(1, args.nr_gpu):
            for j in range(len(grads[0])):
                grads[0][j] += grads[i][j]

        train_step = adam_updates(all_params, grads[0], lr=args.learning_rate)

def make_feed_dict(data, is_training=True, dropout_p=0.5):
    ds = np.split(data, args.nr_gpu)
    feed_dict = {is_trainings[i]: is_training for i in range(args.nr_gpu)}
    feed_dict.update({dropout_ps[i]: dropout_p for i in range(args.nr_gpu)})
    feed_dict.update({ xs[i]:ds[i] for i in range(args.nr_gpu) })
    return feed_dict


initializer = tf.global_variables_initializer()
saver = tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:

    sess.run(initializer)
    data = next(train_set)[0][:,:,:,None]
    feed_dict = make_feed_dict(data)
    max_num_epoch = 200
    for epoch in range(max_num_epoch+1):
        print(epoch, "........")
        tt = time.time()
        for data in train_set:
            data = data[0][:, :, :, None]
            feed_dict = make_feed_dict(data, is_training=True, dropout_p=0.5)
            sess.run(train_step, feed_dict=feed_dict)

        ls = []
        for data in val_set:
            data = data[0][:, :, :, None]
            feed_dict = make_feed_dict(data, is_training=False, dropout_p=0.)
            l = sess.run(models[0].loss, feed_dict=feed_dict)
            print(sess.run(models[0].test, feed_dict=feed_dict))
            ls.append(l)
        print(np.mean(ls))
