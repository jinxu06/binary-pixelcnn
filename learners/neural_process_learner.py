import sys
import random
import numpy as np
import tensorflow as tf
from learners.learner import Learner
from blocks.optimizers import adam_updates
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from blocks.plots import sort_x
# from blocks.plots import visualize_func
# from data.dataset import Dataset



class NPLearner(Learner):

    def __init__(self, session, parallel_models, optimize_op, train_set=None, eval_set=None, variables=None, lr=0.001):
        super().__init__(session, parallel_models, optimize_op, train_set, eval_set, variables)
        self.lr = lr

        grads = []
        for i in range(self.nr_model):
            grads.append(self.parallel_models[i].grads)
        with tf.device('/gpu:0'):
            for i in range(1, self.nr_model):
                for j in range(len(grads[0])):
                    grads[0][j] += grads[i][j]
        self.aggregated_grads = grads[0]
        self.optimize_op = adam_updates(tf.trainable_variables(), self.aggregated_grads, lr=lr)

    def set_session(self, sess):
        self.session = sess

    def get_session(self):
        return self.session

    def train(self, meta_batch, num_shots, test_shots):
        assert meta_batch==self.nr_model, "nr_model != meta_batch"
        tasks = self.train_set.sample(meta_batch)
        feed_dict = {}
        for i, task in enumerate(tasks):
            X_value, y_value = task.sample(num_shots+test_shots)
            X_c_value, X_t_value = X_value[:num_shots], X_value[num_shots:]
            y_c_value, y_t_value = y_value[:num_shots], y_value[num_shots:]
            feed_dict.update({
                self.parallel_models[i].X_c: X_c_value,
                self.parallel_models[i].y_c: y_c_value,
                self.parallel_models[i].X_t: X_t_value,
                self.parallel_models[i].y_t: y_t_value,
                self.parallel_models[i].is_training: True,
            })
        self.get_session().run(self.optimize_op, feed_dict=feed_dict)


    def evaluate(self, eval_samples, num_shots, test_shots):
        ls = []
        for _ in range(eval_samples):
            X_value, y_value = self.eval_set.sample(1)[0].sample(num_shots+test_shots)
            X_c_value, X_t_value = X_value[:num_shots], X_value[num_shots:]
            y_c_value, y_t_value = y_value[:num_shots], y_value[num_shots:]
            l = [m.compute_loss(self.get_session(), X_c_value, y_c_value, X_t_value, y_t_value, is_training=False) for m in self.parallel_models]
            ls.append(l)
        return np.mean(ls)

    def test(self, num_function, num_shots, test_shots):
        fig = plt.figure(figsize=(10,6))
        a = int(np.sqrt(num_function))
        for i in range(num_function):
            ax = fig.add_subplot(a,a,i+1)
            X_value, y_value = self.eval_set.sample(1)[0].sample(num_shots+test_shots)
            X_c_value, X_t_value = X_value[:num_shots], X_value[num_shots:]
            y_c_value, y_t_value = y_value[:num_shots], y_value[num_shots:]
            m = self.parallel_models[0]
            ax.plot(*sort_x(X_value[:,0], y_value), "+-")
            for k in range(20):
                X_eval = np.linspace(-2., 2., num=100)[:,None]
                y_hat = m.predict(self.session, X_c_value, y_c_value, X_eval)
                ax.plot(X_eval[:,0], y_hat, "-", color='gray', alpha=0.3)
                #ax.scatter(X_t_value[:,0], y_hat)
        #plt.show()
        fig.savefig("results/np1.pdf")
        plt.close()


    def run(self, num_epoch, eval_interval, save_interval, eval_samples, meta_batch, num_shots, test_shots):

        for epoch in range(1, num_epoch+1):
            self.qclock()
            for k in range(100):
                self.train(meta_batch, num_shots, test_shots)
            train_time = self.qclock()
            if epoch % eval_interval == 0:
                v = self.evaluate(eval_samples, num_shots, test_shots)
            print("Epoch {0}: {1:0.3f}s ...................".format(epoch, train_time))
            print("    Eval Loss: ", v)
            sys.stdout.flush()
            if epoch % save_interval == 0:
                self.test(9, num_shots, test_shots)
