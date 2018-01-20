"""
Expected improvement acquisition.
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
    """
    Preprocess preference data for use in model.

    Parameters
    ----------
    data : PreferenceDict
        A PreferenceDict object.

    Returns
    -------
    X : np.array, shape (N, D)
        The preference items.
    y : dict
        Mapping from indices corresponding to items (rows) in X to preferences.
        For instance, {(0, 1): 1} means that X[0] is preferred over X[1].
    """
    preferences = sorted(data.preferences())
    ix = {k: v for v, k in enumerate(preferences)}
    X = np.array(preferences)
    y = {(ix[a], ix[b]): v for (a, b), v in data.items()}
    return X, y


def expected_improvement(mu, std, best_f):
    """
    Compute the expected improvement at given point(s).

    Parameters
    ----------
    mu : float or np.array
        Mean of Gaussian process at given point(s).
    std : float or np.array
        Standard deviation of Gaussian process at given point(s).
    best_f : float
        Current maximum.

    Returns
    -------
    ei : float or np.array
        Expected improvement at given point(s).
    """
    mu = np.array(mu)
    std = np.array(std)
    if (std < 0).any():
        raise ValueError("stddev cannot be negative: {}".format(std))

    mask = std > 0
    d = np.where(mask, (mu - best_f) / std, 0.0)
    z = norm()
    return np.where(mask, (mu - best_f) * z.cdf(d) + std * z.pdf(d), 0.0)


class ExpectedImprovementAcquirer(Acquirer):
    """
    Expected improvement acquirer for Gaussian process model.

    Parameters
    ----------
    data : PreferenceDict
        A PreferenceDict object.
    model : PreferenceModel
        A PreferenceModel object.
    optimizer : AcquisitionOptimizer
        An AcquisitionOptimizer object.
    """

    def __init__(self, data, model, optimizer):
        self.data = data
        self.model = model
        self.optimizer = optimizer

        self._next = None
        self._best = None
        self._valuations = None

    @property
    def next(self):
        if self._next is None:
            # preprocess data
            X, y = preprocess_data(self.data)

            # infer posterior
            self.model.fit(X, y)

            # determine current best
            f = self.model.mean(X)
            self._valuations = [(tuple(x), fx) for x, fx in zip(X, f)]
            best_ix = np.argmax(f)
            best_f = f[best_ix]
            self._best = tuple(X[best_ix])

            # find highest-value item
            def func(x):
                if x.ndim == 1:
                    x = [x]
                mean = self.model.mean(x)
                std = np.sqrt(self.model.variance(x))
                return expected_improvement(mean, std, best_f)

            self._next = tuple(self.optimizer.maximize(func))

        return self._next

    @property
    def best(self):
        if self._best is None:
            msg = "call `next` before `best`"
            raise ValueError(msg)
        return self._best

    @property
    def valuations(self):
        if self._valuations is None:
            msg = "call `next` before `valuations`"
            raise ValueError(msg)
        return self._valuations

    def update(self, r, c, preference):
        # reset `next`
        self._next = None

        # add new preference
        self.data[r, c] = preference
