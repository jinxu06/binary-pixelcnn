import random
import numpy as np
import tensorflow as tf
from learners.learner import Learner
import matplotlib.pyplot as plt
from blocks.plots import visualize_func
from data.dataset import Dataset


class NPLearner(Learner):

    def __init__(self, session, parallel_models, optimize_op, train_set=None, eval_set=None, variables=None):
        super().__init__(session, parallel_models, optimize_op, train_set, eval_set, variables)

    def _data_preprocessing(self, data):
        return data

    def _make_feed_dict(self, data, is_training=True, z_value=None, use_z_ph=False):
        data = self._data_preprocessing(data)
        X, y = data
        X_c, X_t = np.split(X, 2)
        y_c, y_t = np.split(y, 2)
        Xs_c = np.split(X_c, self.nr_model)
        ys_c = np.split(y_c, self.nr_model)
        Xs_t = np.split(X_t, self.nr_model)
        ys_t = np.split(y_t, self.nr_model)
        feed_dict = {}
        feed_dict.update({m.is_training: is_training for m in self.parallel_models})
        # feed_dict.update({m.X: Xs[i] for i, m in enumerate(self.parallel_models)})
        # feed_dict.update({m.y: ys[i] for i, m in enumerate(self.parallel_models)})
        feed_dict.update({m.X_c: Xs_c[i] for i, m in enumerate(self.parallel_models)})
        feed_dict.update({m.y_c: ys_c[i] for i, m in enumerate(self.parallel_models)})
        feed_dict.update({m.X_t: Xs_t[i] for i, m in enumerate(self.parallel_models)})
        feed_dict.update({m.y_t: ys_t[i] for i, m in enumerate(self.parallel_models)})
        if use_z_ph:
            feed_dict.update({m.use_z_ph: True for m in self.parallel_models})
            feed_dict.update({m.z_ph: z_value for m in self.parallel_models})
        return feed_dict

    def train_epoch(self):
        for k in range(100):
            sines= self.train_set.sample(1)
            samples = sines[0].sample(200)
            train_set, val_set = samples[:100], samples[100:]
            train_set = Dataset(batch_size=100, X=train_set[:, 0:1], y=train_set[:, 1])
            val_set = Dataset(batch_size=100, X=val_set[:, 0:1], y=val_set[:, 1])
            data = next(train_set)
            feed_dict = self._make_feed_dict(data, is_training=True)
            self.session.run(self.optimize_op, feed_dict=feed_dict)

    def evaluate(self):
        ls = []
        for k in range(100):
            sines= self.eval_set.sample(1)
            samples = sines[0].sample(200)
            train_set, val_set = samples[:100], samples[100:]
            train_set = Dataset(batch_size=100, X=train_set[:, 0:1], y=train_set[:, 1])
            val_set = Dataset(batch_size=100, X=val_set[:, 0:1], y=val_set[:, 1])
            data = next(train_set)
            feed_dict = self._make_feed_dict(data, is_training=False)
            l = self.session.run([m.loss for m in self.parallel_models], feed_dict=feed_dict)
            ls.append(l)
        return np.mean(ls)


    def predict(self, sine=None, z_value=None):
        if sine is None:
            sine= self.eval_set.sample(1)[0]
        samples = sine.sample(2000)
        train_set, val_set = samples[:1000], samples[1000:]
        train_set = Dataset(batch_size=100, X=train_set[:, 0:1], y=train_set[:, 1])
        val_set = Dataset(batch_size=100, X=val_set[:, 0:1], y=val_set[:, 1])
        Xs, ys, ps = [], [], []
        for data in train_set:
            if z_value is None:
                feed_dict = self._make_feed_dict(data, is_training=False)
            else:
                feed_dict = self._make_feed_dict(data, is_training=False, z_value=z_value, use_z_ph=True)
            data = self._data_preprocessing(data)
            X, y = data
            p = self.session.run([m.predictions for m in self.parallel_models], feed_dict=feed_dict)
            Xs.append(X)
            ys.append(y)
            ps += p
        Xs = np.concatenate(Xs, axis=0)
        ys = np.concatenate(ys, axis=0)
        ps = np.concatenate(ps, axis=0)
        return Xs, ys, ps

    def _test(self):
        print("testing ......")
        ls = []
        for k in range(1):
            sines= self.eval_set.sample(1)
            samples = sines[0].sample(200)
            train_set, val_set = samples[:100], samples[100:]
            train_set = Dataset(batch_size=100, X=train_set[:, 0:1], y=train_set[:, 1])
            val_set = Dataset(batch_size=100, X=val_set[:, 0:1], y=val_set[:, 1])
            data = next(train_set)
            feed_dict = self._make_feed_dict(data, is_training=False)
            mean = self.session.run([m.z_mu for m in self.parallel_models], feed_dict=feed_dict)
            std = self.session.run([m.z_sigma for m in self.parallel_models], feed_dict=feed_dict)
            print(mean)
            print(std)


    def run(self, num_epoch, eval_interval, save_interval):

        for epoch in range(1, num_epoch+1):
            self.qclock()
            self.train_epoch()
            train_time = self.qclock()
            if epoch % eval_interval == 0:
                v = self.evaluate()
                #self._test()
            if epoch % save_interval == 0:
                sine = self.eval_set.sample(1)[0]

                Xs, ys, ps = self.predict(sine, z_value=None)
                ax = visualize_func(Xs, ys, ax=None)
                ax = visualize_func(Xs, ps, ax=ax)

                z_value = np.ones((1, 10))
                Xs, ys, ps = self.predict(sine, z_value=z_value)
                ax = visualize_func(Xs, ps, ax=ax)

                z_value = - np.ones((1, 10))
                Xs, ys, ps = self.predict(sine, z_value=z_value)
                ax = visualize_func(Xs, ps, ax=ax)

                plt.show()
            print("Epoch {0}: {1:0.3f}s ...................".format(epoch, train_time))
            print("    Eval Loss: ", v)
