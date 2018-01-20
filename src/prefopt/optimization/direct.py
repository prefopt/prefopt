"""
DIRECT optimizer.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from scipydirect import minimize

from prefopt.acquisition import Optimizer
from prefopt.optimization import OptimizationError

__all__ = ['DirectOptimizer']


class DirectOptimizer(Optimizer):
    """
    DIRECT optimizer.

    Parameters
    ----------
    bounds : array-like
        Bounds of the form (lower, upper) for each dimension of x.
    algmethod : int, optional (default: 1)
        Whether to use the original DIRECT algorithm (algmethod=0) or the
        modified version (algmethod=1).
    """

    def __init__(self, bounds, algmethod=1, **kwargs):
        kwargs.update(algmethod=algmethod)
        self._minimize = functools.partial(minimize, bounds=bounds, **kwargs)

    def maximize(self, func):
        res = self._minimize(lambda x: -func(x))
        if not res.success:
            raise OptimizationError(res.message)
        return res.x
