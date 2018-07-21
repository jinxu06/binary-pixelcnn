#!/usr/bin/python
import numpy as np
from data.gpsampler import GPSampler


gpsampler = GPSampler(input_range=[-2., 2.], var_range=[0.5, 0.5], max_num_samples=200)

gpfuncs = []
for k in range(600):
    print(k)
    gpfuncs += gpsampler.sample(100)

xs, ys = [], []
for f in gpfuncs:
    x, y = f.get_all_samples()
    xs.append(x)
    ys.append(y)
xs = np.array(xs)
ys = np.array(ys)

print(xs.shape)
print(ys.shape)

np.savez("gpsamples_var05", xs=xs, ys=ys)
data = np.load("gpsamples_var05.npz")
xs, ys = data["xs"], data["ys"]

print(xs.shape)
print(ys.shape)
# import matplotlib.pyplot as plt
# plt.style.use("ggplot")
# plt_vals = []
# for f in gpfuncs:
#     x, y = f.get_all_samples()
#     x = x[:, 0]
#     p = np.argsort(x)
#     x = x[p]
#     y = y[p]
#     plt_vals.extend([x, y, "-"])
# plt.plot(*plt_vals)
# plt.show()
