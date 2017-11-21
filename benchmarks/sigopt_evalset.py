#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Runner for the SigOpt `evalset` benchmark suite for black-box optimization.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from evalset.test_funcs import CosineMixture
from prefopt.acquisition import ExpectedImprovementAcquirer
from prefopt.data import UniformPreferenceDict
from prefopt.experiment import (
    InputPresenter,
    OutputPresenter,
    PreferenceExperiment
)
from prefopt.model.thurstone_mosteller import (
    ThurstoneMostellerGaussianProcessModel
)
from prefopt.optimization import DirectOptimizer


class BenchmarkInputPresenter(InputPresenter):
    """
    Presenter for benchmark test functions.

    The presenter maps the latent test function to a preference relation.

    Paramters
    ---------
    func : function
        Test function to be minimized (i.e., lower is better).
    """

    def __init__(self, func):
        self.func = func

    def get_choice(self, a, b):
        a = np.array(a)
        b = np.array(b)
        return 1 if self.func.do_evaluate(a) < self.func.do_evaluate(b) else -1


class BenchmarkOutputPresenter(OutputPresenter):
    """
    Presenter for benchmark output.

    Parameters
    ----------
    func : str
        Name of the function that is being optimized.
    """

    FMT = '{func}: iteration {i}: {xn} vs {xb} -> {choice}'

    def __init__(self, func):
        self.func = func

    def present(self, i, xn, xb, choice):
        msg = self.FMT.format(
            func=self.func,
            i=i,
            xn=xn,
            xb=xb,
            choice=choice
        )
        print(msg)


def sample_uniformly(bounds):
    """
    Sample uniformly from hyperrectangle.

    Parameters
    ----------
    bounds : iterable
        Tuples with (lower, upper) bounds for each dimension.

    Returns
    -------
    x : tuple
        Uniform sample from hyperrectangle.

    Examples
    --------
    >>> bounds = [(0, 1), (-3, -1), (-2, 3)]
    >>> x = sample_uniformly(bounds)
    >>> all(a <= u < b for (a, b), u in zip(bounds, x))
    True
    """
    t = np.random.uniform(size=len(bounds))
    x = tuple(a + u * (b - a) for (a, b), u in zip(bounds, t))
    return x


def run():
    # set RNG seed
    np.random.seed(0)
    tf.set_random_seed(0)

    # set up
    func = CosineMixture()
    bounds = func.bounds

    presenter = BenchmarkInputPresenter(func)
    output = BenchmarkOutputPresenter(func.__class__.__name__)

    optimizer = DirectOptimizer(bounds)
    model = ThurstoneMostellerGaussianProcessModel()
    data = UniformPreferenceDict(2)

    acquirer = ExpectedImprovementAcquirer(
        data,
        model,
        optimizer
    )

    ex = PreferenceExperiment(
        acquirer,
        presenter,
        output
    )

    # add initial observation
    n_initial_samples = 3
    for _ in range(n_initial_samples):
        x1 = sample_uniformly(bounds)
        x2 = sample_uniformly(bounds)
        data[x1, x2] = presenter.get_choice(x1, x2)

    # run experiment
    n_iter = 3
    for i in range(n_iter):
        ex.run()
