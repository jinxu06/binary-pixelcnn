import sys
import random
import numpy as np
import tensorflow as tf
from learners.learner import Learner
from blocks.optimizers import adam_updates
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from blocks.plots import sort_x
# from blocks.plots import visualize_func
# from data.dataset import Dataset



class MAMLLearner(Learner):

    def __init__(self, session, parallel_models, optimize_op, train_set=None, eval_set=None, variables=None, lr=0.001, device_type='gpu', save_dir="test"):
        super().__init__(session, parallel_models, optimize_op, train_set, eval_set, variables)
        self.lr = lr
        self.save_dir = save_dir

        grads = []
        for i in range(self.nr_model):
            grads.append(self.parallel_models[i].grads)
        with tf.device('/' + device_type + ':0'):
            for i in range(1, self.nr_model):
                for j in range(len(grads[0])):
                    grads[0][j] += grads[i][j]
        self.aggregated_grads = grads[0]

        self.optimize_op = adam_updates(variables, self.aggregated_grads, lr=self.lr)

    def set_session(self, sess):
        self.session = sess

    def get_session(self):
        return self.session

    def train(self, meta_batch, num_shots, test_shots):
        assert meta_batch==self.nr_model, "nr_model != meta_batch"
        tasks = self.train_set.sample(meta_batch)
        feed_dict = {}
        for i, task in enumerate(tasks):

            num_shots = np.random.randint(low=1, high=30)
            test_shots = np.random.randint(low=1, high=10)
            num_shots, test_shots = 10, 10

            X_value, y_value = task.sample(num_shots+test_shots)
            X_c_value, X_t_value = X_value[:num_shots], X_value[num_shots:]
            y_c_value, y_t_value = y_value[:num_shots], y_value[num_shots:]
            feed_dict.update({
                self.parallel_models[i].X_c: X_c_value,
                self.parallel_models[i].y_c: y_c_value,
                self.parallel_models[i].X_t: X_value,
                self.parallel_models[i].y_t: y_value,
                self.parallel_models[i].is_training: True,
            })
        self.get_session().run(self.optimize_op, feed_dict=feed_dict)



    def evaluate(self, eval_samples, num_shots, test_shots):
        ls = []
        for _ in range(eval_samples):

            num_shots = np.random.randint(low=1, high=30)
            test_shots = np.random.randint(low=1, high=10)
            num_shots, test_shots = 10, 10

            X_value, y_value = self.eval_set.sample(1)[0].sample(num_shots+test_shots)
            X_c_value, X_t_value = X_value[:num_shots], X_value[num_shots:]
            y_c_value, y_t_value = y_value[:num_shots], y_value[num_shots:]
            l = [m.compute_loss(self.get_session(), X_c_value, y_c_value, X_value, y_value, is_training=False) for m in self.parallel_models]
            ls.append(l)
        return np.mean(ls)

    def test(self, num_function, num_shots, test_shots, epoch=1, input_range=(-2., 2.)):
        fig = plt.figure(figsize=(10,10))
        # a = int(np.sqrt(num_function))
        for i in range(num_function):
            # ax = fig.add_subplot(a,a,i+1)
            ax = fig.add_subplot(4,3,i+1)
            sampler = self.eval_set.sample(1)[0]

            c = [1, 4, 8, 16, 32, 64]
            num_shots = c[(i%6)]

            X_value, y_value = sampler.sample(num_shots+test_shots)
            X_c_value, X_t_value = X_value[:num_shots], X_value[num_shots:]
            y_c_value, y_t_value = y_value[:num_shots], y_value[num_shots:]
            m = self.parallel_models[0]
            X_gt, y_gt = sampler.get_all_samples()
            ax.plot(*sort_x(X_gt[:,0], y_gt), "-")
            ax.scatter(X_c_value[:,0], y_c_value)


            X_eval = np.linspace(self.eval_set.input_range[0], self.eval_set.input_range[1], num=100)[:,None]
            # step 1
            y_hat = m.predict(self.session, X_c_value, y_c_value, X_eval, step=1)
            ax.plot(X_eval[:,0], y_hat, ":", color='gray', alpha=0.3)
            # step 5
            y_hat = m.predict(self.session, X_c_value, y_c_value, X_eval, step=5)
            ax.plot(X_eval[:,0], y_hat, "--", color='gray', alpha=0.3)
            # step 10
            y_hat = m.predict(self.session, X_c_value, y_c_value, X_eval, step=10)
            ax.plot(X_eval[:,0], y_hat, "-", color='gray', alpha=0.3)

        fig.savefig("figs/maml-{0}-{1}.pdf".format(self.eval_set.dataset_name, epoch))
        plt.close()

    def run_eval(self, num_func, num_shots=1, test_shots=50):
        m = self.parallel_models[0]
        saver = tf.train.Saver(var_list=self.variables)
        ckpt_file = self.save_dir + '/params.ckpt'
        print('restoring parameters from', ckpt_file)
        saver.restore(self.session, ckpt_file)
        evals = []
        for _ in range(num_func):
            sampler = self.eval_set.sample(1)[0]
            X_value, y_value = sampler.sample(num_shots+test_shots)
            X_c_value, X_t_value = X_value[:num_shots], X_value[num_shots:]
            y_c_value, y_t_value = y_value[:num_shots], y_value[num_shots:]
            y_t_hat = m.predict(self.session, X_c_value, y_c_value, X_t_value, step=10)
            evals.append(np.mean(np.pow(y_t_value - y_t_hat, 2)))
        eval = np.mean(evals)
        print(".......... EVAL: {} ............".format(eval))


    def run(self, num_epoch, eval_interval, save_interval, eval_samples, meta_batch, num_shots, test_shots, load_params=False):
        num_figures = 12
        saver = tf.train.Saver(var_list=self.variables)

        if load_params:
            ckpt_file = self.save_dir + '/params.ckpt'
            print('restoring parameters from', ckpt_file)
            saver.restore(self.session, ckpt_file)

        self.test(num_figures, num_shots, test_shots, epoch=0)

        for epoch in range(1, num_epoch+1):

            self.qclock()
            for k in range(1000):
                self.train(meta_batch, num_shots, test_shots)
            train_time = self.qclock()
            print("Epoch {0}: {1:0.3f}s ...................".format(epoch, train_time))
            if epoch % eval_interval == 0:
                v = self.evaluate(eval_samples, num_shots, test_shots)
                print("    Eval Loss: ", v)
            if epoch % save_interval == 0:
                print("\tsave figure")
                self.test(num_figures, num_shots, test_shots, epoch=epoch)
                print("\tsave checkpoint")
                saver.save(self.session, self.save_dir + '/params.ckpt')
            sys.stdout.flush()
