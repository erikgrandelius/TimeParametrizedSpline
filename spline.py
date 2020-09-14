import numpy as np

class TimeParametrizedSpline(object):
    """
    Implements a time parametrized Spline curve. From a time series, (t_i,x_i), t in [0,T] x
    a time parametrized spline s is fitted, with s(t_i) = x_i, up to a given tolerance.
    """
    def __init__(self, control_points, para):
        #the Control points of the spline. A purely geometric notion
        self.c = control_points
        #time parametrization, callable function from p :[0,1] -> [0,k]
        self.u = para

    def interpolate(self, t, x):
        return splrep

    def findMaxError(self, t, x):
        return argmax()

    def __call__(self, t):
        #knot indicies

        return np.cos(t)

def fit(t,x, tol=1e-5):

    s = TimeParametrizedSpline(2,3)
        return s.c, s.p

def deBoor(k: int, x: int, t, c, p: int):
    """Evaluates S(x).

    Arguments
    ---------
    k: Index of knot interval that contains x.
    x: Position.
    t: Array of knot positions, needs to be padded as described above.
    c: Array of control points.
    p: Degree of B-spline.
    """
    d = [c[j + k - p] for j in range(0, p + 1)]

    for r in range(1, p + 1):
        for j in range(p, r - 1, -1):
            alpha = (x - t[j + k - p]) / (t[j + 1 + k - r] - t[j + k - p])
            d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j]

    return d[p]

a = np.array([1,2])
spline = Spline(a)
