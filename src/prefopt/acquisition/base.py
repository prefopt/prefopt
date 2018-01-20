"""
Acquisition for active preference learning.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod, abstractproperty

__all__ = [
    'Acquirer',
    'PreferenceModel',
    'Optimizer',
]


class PreferenceModelMeta(object):
    """Preference model metaclass."""

    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, X, y):
        """Perform inference in the preference model."""

    @abstractmethod
    def mean(self, x):
        """The mean function evaluated at point x."""

    @abstractmethod
    def variance(self, x):
        """The variance function evaluated at point x."""


class PreferenceModel(PreferenceModelMeta):
    """Companion class for PreferenceModelMeta."""


class OptimizerMeta(object):
    """Metaclass for acquisition function optimizer."""

    __metaclass__ = ABCMeta

    @abstractmethod
    def maximize(self, func):
        """Maximize the objective function over the bounded domain."""


class Optimizer(OptimizerMeta):
    """Companion class for OptimizerMeta."""


class AcquirerMeta(object):
    """
    Acquirer metaclass.

    This is where all the state associated with a preference experiment lives.
    """

    __metaclass__ = ABCMeta

    @abstractproperty
    def next(self):
        """Next query point."""

    @abstractproperty
    def best(self):
        """Current best point."""

    @abstractproperty
    def valuations(self):
        """Current valuations."""

    @abstractmethod
    def update(self, r, c, preference):
        """Update acquirer with new preference."""


class Acquirer(AcquirerMeta):
    """Companion class for AcquirerMeta."""
