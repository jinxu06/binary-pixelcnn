import os
import sys
import json
import argparse
import time
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from blocks.helpers import visualize_samples, get_nonlinearity, int_shape, get_trainable_variables
from blocks.optimizers import adam_updates
import data.mnist as mnist
from models.binary_pixelcnn import BinaryPixelCNN
from blocks.plots import visualize_samples
from data.omniglot import OmniglotDataSource, Omniglot
from data.dataset import Dataset
from learners.learner import Learner

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--debug', help='', action='store_true', default=False)
parser.add_argument('-mne', '--max_num_epoch', type=int, default=200, help="maximal number of epoches")
parser.add_argument('-ds', '--data_set', type=str, default="omniglot", help='dataset name')
parser.add_argument('-is', '--img_size', type=int, default=28, help="size of input image")
parser.add_argument('-bs', '--batch_size', type=int, default=100, help='Batch size during training per GPU')
parser.add_argument('-ng', '--nr_gpu', type=int, default=1, help='How many GPUs to distribute the training across?')
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


source = OmniglotDataSource("/data/ziz/not-backed-up/jxu/omniglot")
source.split_train_test(1200)
omniglot = Omniglot(source.test_set, inner_batch_size=20)
test_data, _ = omniglot.sample_mini_dataset(num_classes=1, num_shots=20, test_shots=0)

# train_data, _ = omniglot.sample_mini_dataset(num_classes=1200 * 4, num_shots=20, test_shots=0)
# all_data = []
# for d in train_data:
#     d = d[0]#[:, :, :, None]
#     d = 1 - d
#     all_data.append(d)
# all_data = np.concatenate(all_data, axis=0)
# np.random.shuffle(all_data)
# train_set, val_set = all_data[:50000], all_data[50000:60000]
# np.savez("omniglot", train=train_set, val=val_set)

# data = np.load("omniglot.npz")
# train_set, val_set = data['train'], data['val']
# train_set = Dataset(batch_size=args.batch_size * args.nr_gpu, X=train_set)
# val_set = Dataset(batch_size=args.batch_size * args.nr_gpu, X=val_set)

args.batch_size = 10
args.save_interval = 1
all_data = []
for d in test_data:
    d = d[0]#[:, :, :, None]
    d = 1 - d
    all_data.append(d)
all_data = np.concatenate(all_data, axis=0)
np.random.shuffle(all_data)
train_set, val_set = all_data[:10], all_data[10:]
train_set = Dataset(batch_size=args.batch_size * args.nr_gpu, X=train_set)
val_set = Dataset(batch_size=args.batch_size * args.nr_gpu, X=val_set)


xs = [tf.placeholder(tf.float32, shape=(args.batch_size, args.img_size, args.img_size, 1)) for i in range(args.nr_gpu)]
is_trainings = [tf.placeholder(tf.bool, shape=()) for i in range(args.nr_gpu)]
dropout_ps = [tf.placeholder(tf.float32, shape=()) for i in range(args.nr_gpu)]

models = [BinaryPixelCNN(counters={}) for i in range(args.nr_gpu)]
model_opt = {
    "nr_resnet": 3,
    "nr_filters": 30,
    "nonlinearity": tf.nn.elu,
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

learner = Learner(session=None, parallel_models=models, optimize_op=train_step, train_set=train_set, eval_set=val_set, variables=tf.trainable_variables())

# def make_feed_dict(data, is_training=True, dropout_p=0.5):
#     data = np.rint(data)
#     ds = np.split(data, args.nr_gpu)
#     feed_dict = {is_trainings[i]: is_training for i in range(args.nr_gpu)}
#     feed_dict.update({dropout_ps[i]: dropout_p for i in range(args.nr_gpu)})
#     feed_dict.update({ xs[i]:ds[i] for i in range(args.nr_gpu) })
#     return feed_dict
#
# def sample_from_model(sess, data):
#     data = np.rint(data)
#     ds = np.split(data, args.nr_gpu)
#     feed_dict = {is_trainings[i]: False for i in range(args.nr_gpu)}
#     feed_dict.update({dropout_ps[i]: 0. for i in range(args.nr_gpu)})
#     feed_dict.update({ xs[i]:ds[i] for i in range(args.nr_gpu) })
#
#     x_gen = [np.zeros_like(ds[i]) for i in range(args.nr_gpu)]
#     for yi in range(args.img_size):
#         for xi in range(args.img_size):
#             feed_dict.update({xs[i]:x_gen[i] for i in range(args.nr_gpu)})
#             x_hat = sess.run([models[i].x_hat for i in range(args.nr_gpu)], feed_dict=feed_dict)
#             for i in range(args.nr_gpu):
#                 x_gen[i][:, yi, xi, :] = x_hat[i][:, yi, xi, :]
#     return np.concatenate(x_gen, axis=0)


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

    learner.session = sess
    learner.run(num_epoch=args.max_num_epoch, eval_interval=1, save_interval=args.save_interval)

    # max_num_epoch = 200
    # for epoch in range(max_num_epoch+1):
    #     print(epoch, "........")
    #     tt = time.time()
    #     for data in train_set:
    #         data = data[:, :, :, None]
    #         feed_dict = make_feed_dict(data, is_training=True, dropout_p=0.5)
    #         sess.run(train_step, feed_dict=feed_dict)
    #
    #     ls = []
    #     for data in val_set:
    #         data = data[:, :, :, None]
    #         feed_dict = make_feed_dict(data, is_training=False, dropout_p=0.)
    #         l = sess.run([models[i].loss for i in range(args.nr_gpu)], feed_dict=feed_dict)
    #         nats_per_dim = np.mean(l) / (args.img_size**2)
    #         ls.append(nats_per_dim)
    #     print(np.mean(ls))
    #
    #     if epoch % args.save_interval==0:
    #         data = next(val_set)[:, :, :, None]
    #         val_set.reset()
    #         # saver.save(sess, args.save_dir + '/params_' + args.data_set + '.ckpt')
    #         samples = sample_from_model(sess, data)
    #         visualize_samples(data, name="results/gt-{0}.png".format(epoch), layout=(2,5))
    #         visualize_samples(samples, name="results/samples-{0}.png".format(epoch), layout=(2,5))
