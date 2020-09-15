import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


def fitu(t_sample, u_sample, tol=1e-4):
    """
    Fits the arc length parameter u to a cubic spline. If C is the sampled curve
    given by C(t_sample) = x_sample, and S is a spline approximation parametrized
    by normalized arc length u, then C(t_i) = S(u(t_i)).
    ----------
    Parameters:
    -----------
    t_sample:   array_like
                A list of time samples. We do not assume that t_sample is normalized, i.e.
                that t[0] = 0, t[-1] = 1.
    x_sample:   array_like
                A two-dimensional array representing points in the plane.

    Returns:
    --------
    tck:        tuple
                (t,c,k) where t is a knot vector, c is are the control points and k is 3 .
                The knot vector t is padded at the endpoints. The tuple (t,c,k) can be given
                as an argument to scipy.integrate.splev
    """
    T = t_sample[-1] - t_sample[0]
    t_sample = (t_sample - t_sample[0]) / T
    indices = [0, len(t_sample) - 1]
    u = interpolate.splrep(t_sample[indices], u_sample[indices], k=1, s=0 )
    error = abs(interpolate.splev( t_sample, u) - u_sample)

    while np.any(error > tol):
        u = interpolate.splrep(t_sample[indices], u_sample[indices], k=min(3, len(indices) - 1))
        error = abs(interpolate.splev(t_sample, u) - u_sample)
        indices = np.sort(np.append(indices, np.argmax(error)))
    print('Maximum temporal error: ', max(error))
    u = interpolate.splrep(t_sample[indices], u_sample[indices], k=min(3, len(indices) - 1))
    t = np.linspace(0, 1, 1000)
    plt.plot(t_sample, u_sample)
    plt.plot(t, interpolate.splev(t, u))
    plt.scatter(t_sample, u_sample, marker='x')
    plt.scatter(t_sample[indices], u_sample[indices], marker='o')
    plt.legend(['Sample path', 'Cubic fit', 'Sample points', 'Interpolation points'])
    plt.title('Curve length compression')
    plt.show()
    return u

def fitCurve(t_sample, x_sample, y_sample, u_sample, tol = 1e-5):
    """
    Fits sampled curve to a cubic spline. If C is the sampled curve
    given by C(t_sample) = x_sample, and S is a spline approximation parametrized
    by normalized arc length u, then C(t_i) = S(u_).
    ----------
    Parameters:
    -----------
    t_sample:   array_like
                A list of time samples. We do not assume that t_sample is normalized, i.e.
                that t[0] = 0, t[-1] = 1.
    p_sample:   array_like
                A two-dimensional array representing points in the plane.
                p_sample = [|x_0,y_0], .... , [x_n, y_n]]

    Returns:
    --------
    spline :    tuple
                (t,c,k) where t is a knot vector, c is are the control points and k is 3 .
                The knot vector t is padded at the endpoints. The tuple (t,c,k) can be given
                as an argument to scipy.integrate.splev
    """

    indices = [0, len(t_sample) - 1]
    spline, u = interpolate.splprep([x_sample[indices], y_sample[indices]], u = u_sample[indices], k=1, s=0)
    error = np.linalg.norm(interpolate.splev(u_sample, spline) - p_sample.T , axis=0)
    while np.any(error > tol):
        indices = np.sort(np.append(indices, np.argmax(error)))
        spline, u = interpolate.splprep([x_sample[indices], y_sample[indices]], u = u_sample[indices], k=min(len(indices) - 1, 3), s=0)
        error = np.linalg.norm(interpolate.splev(u_sample, spline) - p_sample.T , axis=0)

    print('Max spatial error: ', max(error))
    plt.scatter(x_sample, y_sample, marker='x')
    u_new = np.linspace(0, 1, 1000)
    plt.plot(x_sample, y_sample)
    plt.scatter(x_sample[indices], y_sample[indices], marker='o')
    plt.plot(*interpolate.splev(u_new, spline))
    plt.legend(['Sample path', 'Cubic fit', 'Sample points', 'Interpolation points'])

    plt.title('Spline approximation')
    plt.show()
    return spline

"""
Exampel 1: smooth S-curve
--------
"""
t_sample = np.linspace(0, 1, 200)
x_sample, y_sample =  np.array([np.cos(5 * t_sample**5), 1/(1 + t_sample**5)])
p_sample = np.array([x_sample,y_sample]).T

"""
Exampel 2: V
"""
#t_sample = np.linspace(0, 1, 200)
#x_sample, y_sample = np.array([t_sample**(0.4), abs(t_sample**(.4) - .5)])
#p_sample = np.array([x_sample,y_sample]).T

"""
-------
"""
p_increments = p_sample[1:] - p_sample[:-1]
u_increments = np.concatenate((np.zeros(1), np.linalg.norm(p_increments, axis=1)))
u_sample = np.cumsum(u_increments)
L = u_sample[-1]
u_sample = u_sample / L





spline =fitCurve(t_sample, x_sample, y_sample, u_sample, tol=1e-3)
u = fitu(t_sample, u_sample, tol= 1e-2)


print('Max space-time error: ', np.linalg.norm(interpolate.splev(interpolate.splev(t_sample,u), spline) - p_sample.T))






