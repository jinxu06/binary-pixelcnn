import os
import random
import numpy as np

class DistortedSinusoid(object):

    def __init__(self, sinusoid_generator, noise_generator, noise_level=1.0, dataset_name="sinusoid"):
        self.dataset_name = dataset_name
        assert sinusoid_generator.input_range == noise_generator.input_range, \
                "sinusoid_generator, noise_generator must have some input range"
        self.input_range = sinusoid_generator.input_range
        self.sinusoid_generator = sinusoid_generator
        self.noise_generator = noise_generator
        self.noise_level = noise_level

    def sample(self, num):
        samples_sinusoid = self.sinusoid_generator.sample(num)
        samples_noise = self.noise_generator.sample(num)
        samples = []
        for i in range(num):
            samples.append(DistortedSineWave(samples_sinusoid[i], samples_noise[i], self.noise_level, self.input_range))
        return samples


class DistortedSineWave(object):
    """
    A single sine wave class.
    """
    def __init__(self, samples_sinusoid, samples_noise, noise_level, input_range):
        self.samples_sinusoid = samples_sinusoid
        self.samples_noise = samples_noise
        self.noise_level = noise_level
        self.input_range = input_range

    def sample(self, num_samples):
        xs, ys_noise = self.samples_noise.sample(num_samples)
        _, ys_sinusoid = self.samples_sinusoid.sample(num_samples, xs=xs)
        ys = ys_sinusoid + ys_noise * self.noise_level
        return xs, ys

    def get_all_samples(self):
        return self.sample(self.samples_noise.num_samples)
