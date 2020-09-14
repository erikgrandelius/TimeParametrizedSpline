from scipy.interpolate import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc

# u = np.logspace(0, 1, 100)
sampled_t = np.linspace(0, 1, 30)
sampled_x = np.cos(5 * sampled_t)
sampled_y = 1/(1 + sampled_t)


class SplineFitter:
    def __init__(self, sampled_x, sampled_y, tol=1e-4):
        self.sampled_x = sampled_x
        self.sampled_y = sampled_y
        self.tol = tol


    def fit(self):
        indices = np.array([0, len(self.sampled_x) - 1])
        best_fit, u = splprep([self.sampled_x, self.sampled_y], s=0)
        fit, u1 = splprep([self.sampled_x[indices], self.sampled_y[indices]], k=1, s=0)
        err = np.sum((np.asarray(splev(u, fit)) - np.asarray(splev(u, best_fit))) ** 2, axis=0)
        while np.any(err > self.tol):
            indices = np.sort(np.append(indices, np.argmax(err)))
            fit, u1 = splprep([self.sampled_x[indices], self.sampled_y[indices]], k=min(len(indices) - 1, 3), s=0)
            err = np.sum((np.asarray(splev(u, fit)) - np.asarray(splev(u, best_fit))) ** 2, axis=0)
        print(len(indices))
        plt.plot(*splev(u, fit))


SplineFitter(sampled_x, sampled_y).fit()


spline, u = splprep([sampled_x, sampled_y], s=0)

plt.scatter(sampled_x, sampled_y)
plt.plot(*splev(u, spline))
plt.show()

