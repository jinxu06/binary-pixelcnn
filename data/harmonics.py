import os
import random
import numpy as np


class JXHarmonics(object):

    def __init__(self, input_range=[-6., 6.], dataset_name="jxharmonics"):
        self.dataset_name = dataset_name
        self.input_range = input_range

    def sample(self, num):
        sines = []
        for i in range(num):
            a1 = np.random.uniform(1.0, 5.0)
            a2 = np.random.normal(loc=0., scale=1.)
            b1 = np.random.uniform(0., 2*np.pi)
            b2 = np.random.uniform(0., 2*np.pi)
            omiga = np.random.uniform(5., 7.)
            sines.append(JXHarmonicWave(a1, a2, b1, b2, omiga))
        return sines


class JXHarmonicWave(object):

    def __init__(self, a1, a2, b1, b2, omiga):
        self.a1 = a1
        self.a2 = a2
        self.b1 = b1
        self.b2 = b2
        self.omiga = omiga

    def sample(self, num_samples):
        mu_x = np.random.uniform(-4., 4.)
        xs = np.random.normal(mu_x, 2.0, size=(num_samples,1))
        ys = self.a1 * np.sin(self.omiga * xs + self.b1) + self.a2 * np.sin(2.*self.omiga * xs + self.b2)
        return xs, ys

    def get_all_samples(self):
        return self.sample(200)
