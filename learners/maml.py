import random
import numpy as np
import tensorflow as tf
from blocks.variables import (interpolate_vars, average_vars, subtract_vars, add_vars, scale_vars,
                        VariableState)
from .meta_learner import MetaLearner

class MAML(MetaLearner):

    def __init__(self, session, parallel_models, optimize_op, train_set=None, eval_set=None, variables=None):
        super().__init__(session, parallel_models, optimize_op, train_set, eval_set, variables)

    def _data_preprocessing(self, data):
        if len(data.shape)==3:
            data = data[:, :, :, None]
        data = np.rint(data)
        return data

    def _make_feed_dict(self, data, is_training=True, dropout_p=0.5):
        dd = self._data_preprocessing(data)
        ds = np.split(dd, self.nr_model)
        feed_dict = {}
        feed_dict.update({m.is_training: is_training for m in self.parallel_models})
        feed_dict.update({m.dropout_p: dropout_p for m in self.parallel_models})
        feed_dict.update({m.inputs: ds[i] for i, m in enumerate(self.parallel_models)})
        return feed_dict


    def evaluate(self, num_tasks, num_shots=12, test_shots=8, inner_iter=4, inner_batch_size=4):
        vs = []
        for _ in range(num_tasks):
            train_set, eval_set = self.train_set.sample_mini_dataset(num_classes=1, num_shots=num_shots, test_shots=test_shots)
            train_set.y, eval_set.y = None, None
            old_vars = self._full_state.export_variables()
            for _ in range(inner_iter):
                data = next(train_set)
                train_set.reset()
                feed_dict = self._make_feed_dict(data, is_training=True, dropout_p=0.5)
                self.session.run(self.optimize_op, feed_dict=feed_dict)

            ls = []
            for data in eval_set:
                feed_dict = self._make_feed_dict(data, is_training=False, dropout_p=0.0)
                l = self.session.run([m.loss for m in self.parallel_models], feed_dict=feed_dict)
                nats_per_dim = np.mean(l) / np.prod(data.shape[1:3])
                ls.append(nats_per_dim)
            v = np.mean(ls)

            self._full_state.import_variables(old_vars)
            vs.append(v)
        return np.mean(vs)


    def train_epoch(self, meta_iter_per_epoch, meta_batch_size, meta_step_size, num_shots=12, test_shots=8, inner_iter=4, inner_batch_size=4):




        for _ in range(meta_iter_per_epoch):
            old_vars = self._model_state.export_variables()
            updates = []
            for _ in range(meta_batch_size):
                train_set, eval_set = self.eval_set.sample_mini_dataset(num_classes=1, num_shots=num_shots, test_shots=test_shots)
                train_set.y, eval_set.y = None, None
                for _ in range(inner_iter):
                    batch = next(train_set)
                    train_set.reset()
                    last_backup = self._model_state.export_variables()
                    feed_dict = self._make_feed_dict(batch, is_training=True, dropout_p=0.5)
                    self.session.run(self.optimize_op, feed_dict=feed_dict)
                updates.append(subtract_vars(self._model_state.export_variables(), last_backup))
                self._model_state.import_variables(old_vars)
            update = average_vars(updates)
            self._model_state.import_variables(add_vars(old_vars, scale_vars(update, meta_step_size)))


    def run(self):
        pass 
