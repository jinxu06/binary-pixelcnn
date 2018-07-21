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



class NPLearner(Learner):

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

            total_shots = np.random.randint(low=10, high=50)
            num_shots = np.random.randint(low=8, high=total_shots-1)
            test_shots = total_shots - num_shots
            # test_shots = 0
            # #num_shots, test_shots = 20, 0
            num_shots = np.random.randint(low=1, high=30)
            test_shots = np.random.randint(low=1, high=10)
            # num_shots, test_shots = 20, 10

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

            total_shots = np.random.randint(low=10, high=50)
            num_shots = np.random.randint(low=8, high=total_shots-1)
            test_shots = total_shots - num_shots
            # num_shots, test_shots = 20, 10
            num_shots = np.random.randint(low=1, high=30)
            test_shots = np.random.randint(low=1, high=10)

            X_value, y_value = self.eval_set.sample(1)[0].sample(num_shots+test_shots)
            X_c_value, X_t_value = X_value[:num_shots], X_value[num_shots:]
            y_c_value, y_t_value = y_value[:num_shots], y_value[num_shots:]
            l = [m.compute_loss(self.get_session(), X_c_value, y_c_value, X_value, y_value, is_training=False) for m in self.parallel_models]
            ls.append(l)
        return np.mean(ls)

    def test(self, num_function, num_shots, test_shots, epoch=1):
        fig = plt.figure(figsize=(10,6))
        a = int(np.sqrt(num_function))
        for i in range(num_function):
            ax = fig.add_subplot(a,a,i+1)
            sampler = self.eval_set.sample(1)[0]
            #
            # total_shots = np.random.randint(low=10, high=50)
            # num_shots = np.random.randint(low=total_shots-10, high=total_shots-1)
            # test_shots = total_shots - num_shots
            #num_shots = np.random.randint(low=1, high=30)
            #test_shots = np.random.randint(low=1, high=10)
            # num_shots, test_shots = 20, 10
            num_shots = np.random.randint(low=0, high=5) * 10 + 1
            #
            X_value, y_value = sampler.sample(num_shots+test_shots)
            X_c_value, X_t_value = X_value[:num_shots], X_value[num_shots:]
            y_c_value, y_t_value = y_value[:num_shots], y_value[num_shots:]
            m = self.parallel_models[0]
            X_gt, y_gt = sampler.get_all_samples()
            ax.plot(*sort_x(X_gt[:,0], y_gt), "-")
            ax.scatter(X_c_value[:,0], y_c_value)
            #ax.plot(*sort_x(X_value[:,0], y_value), "+")
            for k in range(20):
                X_eval = np.linspace(-2., 2., num=100)[:,None]
                y_hat = m.predict(self.session, X_c_value, y_c_value, X_eval)
                ax.plot(X_eval[:,0], y_hat, "-", color='gray', alpha=0.3)
                #ax.plot(X_value[:,0], y_hat, "-", color='gray', alpha=0.3)
        fig.savefig("figs/np{0}-1.pdf".format(epoch))
        plt.close()


    def run(self, num_epoch, eval_interval, save_interval, eval_samples, meta_batch, num_shots, test_shots, load_params=False):

        saver = tf.train.Saver(var_list=self.variables)

        if load_params:
            ckpt_file = self.save_dir + '/params.ckpt'
            print('restoring parameters from', ckpt_file)
            saver.restore(sess, ckpt_file)

        self.test(9, num_shots, test_shots, epoch=0)

        for epoch in range(1, num_epoch+1):

            self.qclock()
            for k in range(1000):
                self.train(meta_batch, num_shots, test_shots)
            train_time = self.qclock()
            if epoch % eval_interval == 0:
                v = self.evaluate(eval_samples, num_shots, test_shots)
            print("Epoch {0}: {1:0.3f}s ...................".format(epoch, train_time))
            print("    Eval Loss: ", v)
            if epoch % save_interval == 0:
                print("\tsave figure")
                self.test(9, num_shots, test_shots, epoch=epoch)
                print("\tsave checkpoint")
                saver.save(self.session, self.save_dir + '/params.ckpt')
            sys.stdout.flush()
