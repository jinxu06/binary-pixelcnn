import random
import numpy as np
import tensorflow as tf
from learners.learner import Learner

class RegressionLearner(Learner):

    def __init__(self, session, parallel_models, optimize_op, train_set=None, eval_set=None, variables=None):
        super().__init__(session, parallel_models, optimize_op, train_set, eval_set, variables)

    def _data_preprocessing(self, data):
        return data

    def _make_feed_dict(self, data, is_training=True):
        data = self._data_preprocessing(data)
        X, y = data
        Xs = np.split(X, self.nr_model)
        ys = np.split(y, self.nr_model)
        feed_dict = {}
        feed_dict.update({m.is_training: is_training for m in self.parallel_models})
        feed_dict.update({m.X: Xs[i] for i, m in enumerate(self.parallel_models)})
        feed_dict.update({m.y: ys[i] for i, m in enumerate(self.parallel_models)})
        return feed_dict

    def train_epoch(self):
        for data in self.train_set:
            feed_dict = self._make_feed_dict(data, is_training=True)
            self.session.run(self.optimize_op, feed_dict=feed_dict)

    def evaluate(self):
        ls = []
        for data in self.eval_set:
            feed_dict = self._make_feed_dict(data, is_training=False)
            l = self.session.run([m.loss for m in self.parallel_models], feed_dict=feed_dict)
            ls.append(l)
        return np.mean(ls)

    def predict(self):
        Xs, ys, ps = [], [], []
        for data in self.eval_set:
            feed_dict = self._make_feed_dict(data, is_training=False)
            data = self._data_preprocessing(data)
            X, y = data
            p = self.session.run([m.predictions for m in self.parallel_models], feed_dict=feed_dict)
            Xs.append(X)
            ys.append(y)
            ps += p
        Xs = np.concatenate(Xs, axis=0)
        ys = np.concatenate(ys, axis=0)
        ps = np.concatenate(ps, axis=0)
        print(ys[:10])
        print(ps[:10])


    def run(self, num_epoch, eval_interval, save_interval):
        for epoch in range(1, num_epoch+1):
            self.qclock()
            self.train_epoch()
            train_time = self.qclock()
            if epoch % eval_interval == 0:
                v = self.evaluate()
            if epoch % save_interval == 0:
                self.predict()
            print("Epoch {0}: {1:0.3f}s ...................".format(epoch, train_time))
            print("    Eval Loss: ", v)
