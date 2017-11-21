"""
Expected improvement acquisition for active preference learning.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from scipy.stats import norm

from . import Acquirer

__all__ = [
    'ExpectedImprovementAcquirer',
]


def preprocess_data(data):
    """Preprocess preference data for use in model."""
    preferences = sorted(data.preferences())
    ix = {k: v for v, k in enumerate(preferences)}
    X = np.array(preferences)
    y = {(ix[a], ix[b]): v for (a, b), v in data.items()}
    return X, y


def expected_improvement(mu, std, best_f):
    """
    Compute the expected improvement at given point.

    Parameters
    ----------
    mu : float
        Mean of Gaussian process at given point.
    std : float
        Standard deviation of Gaussian process at given point.
    best_f : float
        Current maximum.

    Returns
    -------
    ei : float
        Expected improvement at given point.
    """
    if std < 0:
        raise ValueError("stddev cannot be negative: {}".format(std))
    elif std == 0:
        return 0.0
    else:
        d = (mu - best_f) / std
        z = norm()
        return (mu - best_f) * z.cdf(d) + std * z.pdf(d)


class ExpectedImprovementAcquirer(Acquirer):
    """
    Expected improvement acquirer for Gaussian process model.

    Parameters
    ----------
    model : PreferenceModel
        PreferenceModel object.
    optimizer : AcquisitionOptimizer
        AcquisitionOptimizer object that can maximize the acquisition function.
    """

    def __init__(self, data, model, optimizer):
        self.data = data
        self.model = model
        self.optimizer = optimizer

        self._next = None
        self._best = None

    @property
    def next(self):
        if self._next is None:
            # preprocess data
            X, y = preprocess_data(self.data)

            # infer posterior
            self.model.fit(X, y)

            # determine current best
            f = self.model.mean(X)
            best_ix = np.argmax(f)
            best_f = f[best_ix]
            self._best = tuple(X[best_ix])

            # find highest-value item
            def func(x):
                mean = self.model.mean([x])
                std = np.sqrt(self.model.variance([x]))
                return expected_improvement(mean, std, best_f)

            self._next = tuple(self.optimizer.maximize(func))

        return self._next

    @property
    def best(self):
        if self._best is None:
            msg = "call `next` before `best`"
            raise ValueError(msg)
        return self._best

    def update(self, r, c, preference):
        # reset `next`
        self._next = None

        # add new preference
        self.data[r, c] = preference
