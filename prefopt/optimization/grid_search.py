"""
Grid search optimizer.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from prefopt.acquisition import Optimizer

import itertools

import numpy as np

__all__ = ['GridSearchOptimizer']


def compute_grid(bounds, n_gridpoints, dtype=np.float32):
    """
    Compute grid points.

    Parameters
    ----------
    bounds : list
        List of bounding boxes of the form (start, end) for each dimension of
        the domain.
    n_gridpoints : float
        Approximate number of grid points.
    dtype : numpy.dtype, optional (default: np.float32)
        The data type of the grid array.

    Returns
    -------
    grid : numpy.array
        Numpy array of grid point tuples.

    Examples
    --------
    >>> bounds = [[0, 10], [0, 10]]
    >>> grid = compute_grid(bounds, 1000)
    >>> len(grid)
    1024
    >>> grid[:3]  # doctest: +NORMALIZE_WHITESPACE
    array([[ 0.        ,  0.        ],
           [ 0.        ,  0.32258064],
           [ 0.        ,  0.64516127]], dtype=float32)
    >>> grid[-3:]  # doctest: +NORMALIZE_WHITESPACE
    array([[ 10.        ,   9.35483837],
           [ 10.        ,   9.67741966],
           [ 10.        ,  10.        ]], dtype=float32)

    """
    # compute the exact number of grid points
    n_dim = len(bounds)
    n_gridpoints_exact = int(np.round(np.power(n_gridpoints, 1.0 / n_dim)))

    # compute gridpoints
    ticks = [np.linspace(*x, num=n_gridpoints_exact) for x in bounds]
    grid = list(itertools.product(*ticks))
    return np.array(grid, dtype=dtype)


class GridSearchOptimizer(Optimizer):
    """
    Grid search optimizer.

    Parameters
    ----------
    bounds : array-like
        Bounds of the form (lower, upper) for each dimension of x.
    n_gridpoints : int, optional (default: 1000000)
        Approximate number of grid points.
    """

    def __init__(self, bounds, n_gridpoints=1000000):
        self.bounds = bounds
        self.n_gridpoints = n_gridpoints

    def maximize(self, func):
        grid = compute_grid(self.bounds, self.n_gridpoints)
        f = func(grid)
        ix = np.argmax(f)
        return grid[ix]
