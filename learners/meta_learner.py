import random
import numpy as np
import tensorflow as tf
from blocks.variables import (interpolate_vars, average_vars, subtract_vars, add_vars, scale_vars,
                        VariableState)

class MetaLearner(object):

    def __init__(self, session, parallel_models, variables=None):
        # transductive, pre_step_op
        self.session = session
        self.parallel_models = parallel_models
        self._model_state = VariableState(self.session, variables or tf.trainable_variables())
        self._full_state = VariableState(self.session,
                                         tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

    def evaluate(self,
                 dataset,
                 input_ph,
                 label_ph,
                 minimize_op,
                 predictions,
                 num_classes,
                 num_shots,
                 inner_batch_size,
                 inner_iters,
                 replacement):
        # train_set, test_set = _split_train_test(
        #     _sample_mini_dataset(dataset, num_classes, num_shots+1))
        train_set, test_set = dataset.sample_task(num_shots+1, num_classes, use_split=True, test_shots=1)
        old_vars = self._full_state.export_variables()
        mini_batches = dataset.mini_batch(train_set, inner_batch_size, inner_iters, replacement)
        for batch in mini_batches:
            inputs, labels = zip(*batch)
            self.session.run(minimize_op, feed_dict={input_ph: inputs, label_ph: labels})
        test_preds = self._test_predictions(train_set, test_set, input_ph, predictions)
        #num_correct = 0 #sum([pred == sample[1] for pred, sample in zip(test_preds, test_set)])
        e = self.eval_metric(test_preds, labels)
        self._full_state.import_variables(old_vars)
        return e

    def _test_predictions(self, train_set, test_set, input_ph, predictions):
        if self._transductive:
            inputs, _ = zip(*test_set)
            return self.session.run(predictions, feed_dict={input_ph: inputs})
        res = []
        for test_sample in test_set:
            inputs, _ = zip(*train_set)
            inputs += (test_sample[0],)
            res.append(self.session.run(predictions, feed_dict={input_ph: inputs})[-1])
        return res
