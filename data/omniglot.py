"""
Loading and augmenting the Omniglot dataset.
To use these APIs, you should prepare a directory that
contains all of the alphabets from both images_background
and images_evaluation.
"""

import os
import random

from PIL import Image
import numpy as np
import tensorflow as tf
import blocks.helpers as helpers
from data.dataset import Dataset

# def _sample_mini_dataset(data_dir, num_classes, num_shots=20):
#     ds = OmniglotDataSource(data_dir=data_dir)
#     ds.split_train_test(1200, augment_train_set=False)
#     cs = ds.sample_classes(num_classes)
#     X = []
#     y = []
#     for idx, c in enumerate(cs):
#         X.append(c.sample(num_shots))
#         y.append(np.array([idx for i in range(num_shots)]))
#     X = np.concatenate(X, axis=0)
#     y = np.concatenate(y, axis=0)
#     return X, y

# def load(data_dir, num_classes, num_shots, batch_size, split, one_hot=True):
#     images, targets = _sample_mini_dataset(data_dir, num_classes)
#     p = np.random.permutation(images.shape[0])
#     images, targets = images[p], targets[p]
#     datasets = []
#     begin = 0
#     for s in split:
#         end = begin + s
#         X = images[begin:end]
#         y = targets[begin:end]
#         X = tf.data.Dataset.from_tensor_slices(X)
#         y = tf.data.Dataset.from_tensor_slices(y)
#         if one_hot:
#             y = y.map(lambda z: tf.one_hot(z, 10))
#         dataset = tf.data.Dataset.zip((X, y)).shuffle(s).batch(batch_size)
#         datasets.append(dataset)
#         begin = end
#     return datasets

def load(data_dir, inner_batch_size, num_train=1200, augment_train_set=False, one_hot=True):
    ds = OmniglotDataSource(data_dir=data_dir)
    ds.split_train_test(num_train, augment_train_set=augment_train_set)
    train_meta_dataset = Omniglot(ds.train_set, inner_batch_size, one_hot)
    test_meta_dataset = Omniglot(ds.test_set, inner_batch_size, one_hot)
    return train_meta_dataset, test_meta_dataset


class Omniglot(object):

    def __init__(self, data_source, inner_batch_size, one_hot=True):
        self.data_source = data_source
        self.inner_batch_size = inner_batch_size
        self.one_hot = one_hot

    # def _load_dataset(self, X, y, num_classes, batch_size, one_hot=True):
    #     num_samples = X.shape[0]
    #     p = np.random.permutation(X.shape[0])
    #     X, y = X[p], y[p]
    #     X = tf.data.Dataset.from_tensor_slices(X)
    #     y = tf.data.Dataset.from_tensor_slices(y)
    #     if one_hot:
    #         y = y.map(lambda z: tf.one_hot(z, num_classes))
    #     dataset = tf.data.Dataset.zip((X, y)).shuffle(num_samples).batch(batch_size)
    #     return dataset


    def sample_mini_dataset(self, num_classes, num_shots, test_shots):
        shuffled = list(self.data_source)
        random.shuffle(shuffled)
        cs = shuffled[:num_classes]
        X = []
        y = []
        X_test = []
        y_test = []
        for idx, c in enumerate(cs):
            inputs = c.sample(num_shots+test_shots)
            targets = np.array([idx for i in range(num_shots+test_shots)])
            X.append(inputs[:num_shots])
            y.append(targets[:num_shots])
            X_test.append(inputs[num_shots:])
            y_test.append(targets[num_shots:])
        X = np.concatenate(X, axis=0)
        y = np.concatenate(y, axis=0)
        X_test = np.concatenate(X_test, axis=0)
        y_test = np.concatenate(y_test, axis=0)

        if self.one_hot:
            y = helpers.one_hot(y, num_classes)
            y_test = helpers.one_hot(y_test, num_classes)

        train_set = Dataset(batch_size=self.inner_batch_size, X=X, y=y, shuffle=True)
        test_set = Dataset(batch_size=self.inner_batch_size, X=X_test, y=y_test, shuffle=False)
        # train_set = self._load_dataset(X, y, num_classes, self.inner_batch_size, self.one_hot)
        # test_set = self._load_dataset(X_test, y_test, num_classes, self.inner_batch_size, self.one_hot)
        return train_set, test_set




class OmniglotDataSource(object):

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data = None

    def _load(self):
        self.data = read_dataset(self.data_dir)

    def split_train_test(self, num_train, augment_train_set=True):
        if self.data is None:
            self._load()
        self.train_set, self.test_set = split_dataset(self.data)
        if augment_train_set:
            self.train_set = list(augment_dataset(self.train_set))
        self.test_set = list(self.test_set)

    def sample_classes(self, num_classes, which_set='train'):
        if which_set == 'train':
            dataset = self.train_set
        elif which_set == 'test':
            dataset = self.test_set
        else:
            raise Exception('which_set is either train or test')
        shuffled = list(dataset)
        random.shuffle(shuffled)
        return shuffled[:num_classes]


def read_dataset(data_dir):
    """
    Iterate over the characters in a data directory.
    Args:
      data_dir: a directory of alphabet directories.
    Returns:
      An iterable over Characters.
    The dataset is unaugmented and not split up into
    training and test sets.
    """
    for alphabet_name in sorted(os.listdir(data_dir)):
        alphabet_dir = os.path.join(data_dir, alphabet_name)
        if not os.path.isdir(alphabet_dir):
            continue
        for char_name in sorted(os.listdir(alphabet_dir)):
            if not char_name.startswith('character'):
                continue
            yield Character(os.path.join(alphabet_dir, char_name), 0, tags={'alphabet': alphabet_name, 'character': char_name, 'rotation': 0})

def split_dataset(dataset, num_train=1200):
    """
    Split the dataset into a training and test set.
    Args:
      dataset: an iterable of Characters.
    Returns:
      A tuple (train, test) of Character sequences.
    """
    all_data = list(dataset)
    random.shuffle(all_data)
    return all_data[:num_train], all_data[num_train:]

def augment_dataset(dataset):
    """
    Augment the dataset by adding 90 degree rotations.
    Args:
      dataset: an iterable of Characters.
    Returns:
      An iterable of augmented Characters.
    """
    for character in dataset:
        for rotation in [0, 90, 180, 270]:
            tags = character.tags
            tags.update({'rotation': rotation})
            yield Character(character.dir_path, rotation=rotation, tags=tags)


class Character:
    """
    A single character class.
    """
    def __init__(self, dir_path, rotation=0, tags={}):
        self.dir_path = dir_path
        self.rotation = rotation
        self._cache = {}
        self.tags = tags.copy()

    def sample(self, num_images):
        """
        Sample images (as numpy arrays) from the class.
        Returns:
          A sequence of 28x28 numpy arrays.
          Each pixel ranges from 0 to 1.
        """
        names = [f for f in os.listdir(self.dir_path) if f.endswith('.png')]
        random.shuffle(names)
        images = []
        for name in names[:num_images]:
            images.append(self._read_image(os.path.join(self.dir_path, name)))
        return images

    def _read_image(self, path):
        if path in self._cache:
            return self._cache[path]
        with open(path, 'rb') as in_file:
            img = Image.open(in_file).resize((28, 28)).rotate(self.rotation)
            self._cache[path] = np.array(img).astype('float32')
            return self._cache[path]
