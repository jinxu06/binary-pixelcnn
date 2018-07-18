#!/usr/bin/python
import numpy as np
from data.gpsampler import GPSampler

gp = GPSampler()
xs, ys = gp.sample(100)


import matplotlib.pyplot as plt
plt.style.use("ggplot")
plt_vals = []
for i in range(0, 5):
    xs, ys = gp.sample(100, [-2, 2], var_range=[0.5,2.])
    p = np.argsort(xs)
    xs = xs[p]
    ys = ys[p]
    plt_vals.extend([xs, ys, "x-"])
plt.plot(*plt_vals)
plt.show()
