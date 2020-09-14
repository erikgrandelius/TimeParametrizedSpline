import numpy as np

class TimeParametrizedSpline(object):
    """
    Implements a time parametrized Spline curve. From a time series, (t_i,x_i),
    a time parametrized spline s is fitted, with s(t_i) = x_i, up to a given tolerance.
    """
    def __init__(self, control_points, para):
        #the Control points of the spline. A purely geometric notion
        self.c = control_points
        #time parametrization, callable function from p :[0,1] -> [0,k]
        self.p = para

    """
    Static method for 
    """
    def __call__(self, t):
        return np.cos(t)
    @staticmethod
    def fit(t , x, y, tol = 1e-5):
        s = TimeParametrizedSpline(2,3)
        return s.c, s.p

    def __call__(self, t ):


a = np.array([1,2])
spline = Spline(a)
