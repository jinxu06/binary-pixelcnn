import tensorflow as tf
import numpy as np
from data.dataset import Dataset
import blocks.helpers as helpers


def load(data_dir, num_classes, batch_size, split=[5./7, 1./7, 1./7], one_hot=True):
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] # np.random.choice(10, num_classes, replace=False).astype(np.int32)
    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata('MNIST original', data_home=data_dir)
    images = mnist['data'].astype(np.float32)
    targets = mnist['target'].astype(np.int32)
    imgs, ts = [], []
    for img, t in zip(images, targets):
        if t in classes:
            imgs.append(img)
            ts.append(t)
    images, targets = np.array(imgs), np.array(ts)
    images /= images.max()
    p = np.random.permutation(images.shape[0])
    images, targets = images[p], targets[p]
    num_samples = images.shape[0]
    split = [int(np.rint(num_samples*s)) for s in split]
    split[-1] += num_samples - np.sum(split)
    datasets = []
    begin = 0
    for s in split:
        end = begin + s
        X = images[begin:end]
        X = np.reshape(X, newshape=(s, 28, 28))
        y = targets[begin:end]
        y -= np.min(y)
        if one_hot:
            y = helpers.one_hot(y, len(classes))
        dataset = Dataset(batch_size=batch_size, X=X, y=y, shuffle=False)
        datasets.append(dataset)
        begin = end
    datasets[0].shuffle = True
    return datasets
