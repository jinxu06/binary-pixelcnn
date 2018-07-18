import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
plt.style.use("ggplot")

def sort_x(x, y):
    p = np.argsort(x)
    return x[p], y[p]

def tile_images(imgs, size=(6, 6)):
    imgs = imgs[:size[0]*size[1], :, :, :]
    if imgs.shape[-1] == 1:
        imgs = np.stack([imgs.copy()[:,:,:,0] for k in range(3)], axis=-1)
    img_h, img_w = imgs.shape[1], imgs.shape[2]
    all_images = np.zeros((img_h*size[0], img_w*size[1], 3), np.uint8)
    for j in range(size[0]):
        for i in range(size[1]):
            all_images[img_h*j:img_h*(j+1), img_w*i:img_w*(i+1), :] = imgs[j*size[1]+i, :, :, :]
    return all_images

def visualize_samples(images, name="results/test.png", layout=[5,5], vrange=[0., 1.]):
    images = (images - vrange[0]) / (vrange[1]-vrange[0]) * 255.
    images = np.rint(images).astype(np.uint8)
    view = tile_images(images, size=layout)
    if name is None:
        return view
    view = Image.fromarray(view, 'RGB')
    view.save(name)

def visualize_func(X, y, ax=None):
    o = X[:, 0].argsort()
    X = X[o]
    y = y[o]
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
    ax.plot(X, y, "+-")
    return ax


# fig = plt.figure(figsize=(8,6))
# ax = fig.add_subplot(111)
#
# for i in range(len(sines)):
#     s = sines[i].sample(500)
#     s = s[s[:,0].argsort()]
#     ax.plot(s[:,0], s[:,1], '.-')
# plt.show()
