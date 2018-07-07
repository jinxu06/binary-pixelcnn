import numpy as np
import tensorflow as tf

class Dataset(object):

    def __init__(self, batch_size, X, y=None, shuffle=True):
        self.init_X = X
        self.init_y = y
        self.size = X.shape[0]
        self.X = X.copy()
        if y is not None:
            self.y = y.copy()
        else:
            self.y = None
        self.cur_pos = 0
        self.shuffle = shuffle
        self.batch_size = batch_size

    def reset(self):
        # self.X = self.init_X.copy()
        # if self.init_y is not None
        #     self.y = self.init_y.copy()
        self.cur_pos = 0

    def _shuffle(self):
        p = np.random.permutation(self.X.shape[0])
        self.X = self.X[p]
        if self.y is not None:
            self.y = self.y[p]

    def _index(self, begin, end):
        if self.y is not None:
            return self.X[begin:end], self.y[begin:end]
        else:
            return self.X[begin:end]


    def __iter__(self):
        return self

    def __next__(self):
        if self.cur_pos == 0 and self.shuffle:
            self._shuffle()
        if self.cur_pos + self.batch_size > self.size:
            self.reset()
            raise StopIteration
        r = self._index(self.cur_pos, self.cur_pos+self.batch_size)
        self.cur_pos += self.batch_size
        return r

    next = __next__
