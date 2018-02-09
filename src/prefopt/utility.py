"""
Utility functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import random
import collections

__all__ = [
    'BoundingBox',
    'NegativeQuadraticUtilityFunction',
]


class BoundingBox(collections.Sequence):

    def __init__(self, bounds):
        if not all(l < u for (l, u) in bounds):
            raise ValueError(
                "lower bound needs to be strictly smaller than upper bound")
        self._bounds = tuple(tuple(x) for x in bounds)

    def __len__(self):
        return len(self._bounds)

    def __getitem__(self, index):
        return self._bounds[index]

    def sample(self):
        if not self._bounds:
            raise ValueError('cannot sample from empty bounding box')
        return tuple(random.uniform(*x) for x in self._bounds)


class UtilityFunctionMeta(object):
    """Utility function metaclass."""

    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def bounds(self):
        """Bounding box of function domain."""

    @abc.abstractproperty
    def argmax(self):
        """Argmax of function on domain."""

    @abc.abstractmethod
    def evaluate(self, x):
        """
        Evaluate utility function.

        Parameters
        ----------
        x : numpy.array
            Point at which to evaluate the function.

        Returns
        -------
        fx : float
            The utility at point x.

        Raises
        ------
        ValueError
            If the point x is outside of the specified bounds.
        """

    def __call__(self, x):
        return self.evaluate(x)


class UtilityFunction(UtilityFunctionMeta):
    """Companion class for UtilityFunctionMeta."""


class NegativeQuadraticUtilityFunction(UtilityFunction):
    """Negative-quadratic utility function."""

    def __init__(self, bounds):
        self.bounds = bounds
        self.lower, self.upper = self.bounds[0]

    @property
    def argmax(self):
        if self.lower > 0:
            return self.lower
        elif self.upper < 0:
            return self.upper
        return 0

    @property
    def bounds(self):
        return self._bounds

    @bounds.setter
    def bounds(self, bounds):
        if len(bounds) > 1:
            raise ValueError('Invalid bounds: {}'.format(bounds))
        self._bounds = bounds

    def evaluate(self, x):
        try:
            x, = x
        except TypeError:
            pass
        if not self.lower <= x <= self.upper:
            raise ValueError("{} outside of bounds {}".format(x, self._bounds))
        return -(x * x)
