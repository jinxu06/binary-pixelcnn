import random
import numpy as np
import tensorflow as tf
from blocks.variables import (interpolate_vars, average_vars, subtract_vars, add_vars, scale_vars,
                        VariableState)
from learners.learner import Learner

class MetaLearner(Learner):

    def __init__(self, session, parallel_models, optimize_op, train_set=None, eval_set=None, variables=None):
        # transductive, pre_step_op
        super().__init__(session, parallel_models, optimize_op, train_set, eval_set, variables)
        self._model_state = VariableState(self.session, variables or tf.trainable_variables())
        self._full_state = VariableState(self.session,
                                         tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))



    def evaluate(self, num_tasks, inner_iter=5):
        vs = []
        for _ in range(num_tasks):
            train_set, eval_set = self.train_set.sample_mini_dataset(num_classes=1, num_shots=10, test_shots=10)
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

    def train_epoch(self, meta_iter, meta_batch_size, meta_step_size, batch_size, inner_iter):
        for _ in range(meta_iter):
            old_vars = self._model_state.export_variables()
            updates = []
            for _ in range(meta_batch_size):
                train_set, eval_set = self.eval_set.sample_mini_dataset(num_classes=1, num_shots=10, test_shots=10)
                train_set.y, eval_set.y = None, None
                task_learner = Learner(self.session, self.parallel_models, self.optimize_op, train_set, eval_set, self.variables)
                for _ in range(inner_iter):
                    batch = next(train_set)
                    train_set.reset()
                    last_backup = self._model_state.export_variables()
                    feed_dict = self._make_feed_dict()
                    self.session.run(optimize_op, feed_dict=self._make_feed_dict(batch))
                updates.append(subtract_vars(self._model_state.export_variables(), last_backup))
                self._model_state.import_variables(old_vars)
            update = average_vars(updates)
            self._model_state.import_variables(add_vars(old_vars, scale_vars(update, meta_step_size)))


    def run(self, num_epoch, eval_interval, save_interval, **kwargs):
        for epoch in range(1, num_epoch+1):
            self.qclock()
            self.train_epoch(meta_iter, meta_batch_size, meta_step_size, batch_size, inner_iter)
            train_time = self.qclock()
            # if epoch % eval_interval == 0:
            v = self.evaluate(num_tasks, num_train_epoch)

            print("Epoch {0}: {1:0.3f}s ...................".format(epoch, train_time))
            print("    Eval Loss: ", v)

    # def evaluate(self,
    #              dataset,
    #              input_ph,
    #              label_ph,
    #              minimize_op,
    #              predictions,
    #              num_classes,
    #              num_shots,
    #              inner_batch_size,
    #              inner_iters,
    #              replacement):
    #     # train_set, test_set = _split_train_test(
    #     #     _sample_mini_dataset(dataset, num_classes, num_shots+1))
    #     train_set, test_set = dataset.sample_task(num_shots+1, num_classes, use_split=True, test_shots=1)
    #     old_vars = self._full_state.export_variables()
    #     mini_batches = dataset.mini_batch(train_set, inner_batch_size, inner_iters, replacement)
    #     for batch in mini_batches:
    #         inputs, labels = zip(*batch)
    #         self.session.run(minimize_op, feed_dict={input_ph: inputs, label_ph: labels})
    #     test_preds = self._test_predictions(train_set, test_set, input_ph, predictions)
    #     #num_correct = 0 #sum([pred == sample[1] for pred, sample in zip(test_preds, test_set)])
    #     e = self.eval_metric(test_preds, labels)
    #     self._full_state.import_variables(old_vars)
    #     return e
    #
    # def _test_predictions(self, train_set, test_set, input_ph, predictions):
    #     if self._transductive:
    #         inputs, _ = zip(*test_set)
    #         return self.session.run(predictions, feed_dict={input_ph: inputs})
    #     res = []
    #     for test_sample in test_set:
    #         inputs, _ = zip(*train_set)
    #         inputs += (test_sample[0],)
    #         res.append(self.session.run(predictions, feed_dict={input_ph: inputs})[-1])
    #     return res
