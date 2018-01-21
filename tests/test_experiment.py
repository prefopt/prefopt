"""
Tests for prefopt.experiment.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import numpy as np
import tensorflow as tf

from prefopt.acquisition import (
    Acquirer,
    ExpectedImprovementAcquirer
)
from prefopt.data import UniformPreferenceDict
from prefopt.experiment import (
    InputPresenter,
    OutputPresenter,
    PreferenceExperiment
)
from prefopt.model import BinaryPreferenceModel
from prefopt.optimization import DirectOptimizer


def verify_preferences(data, presenter):
    for k, v in data.items():
        a, b = k
        choice = presenter.get_choice(a, b)
        if choice != v:
            return False
    return True


class MockAcquirer(Acquirer):

    def __init__(self):
        self.data = UniformPreferenceDict(2)
        self.data[(0, 0), (1, 1)] = 1

        self._best = (0, 0)
        self._next = (2, 2)
        self._valuations = [((0, 0), 1), ((2, 2), 0)]

    @property
    def next(self):
        return self._next

    @property
    def best(self):
        return self._best

    @property
    def valuations(self):
        return self._valuations

    def update(self, r, c, preference):
        self._next = (2, 2)
        self.data[r, c] = preference


class MockInputPresenter(InputPresenter):

    def get_choice(self, a, b):
        # smaller is better
        return 1 if (a < b) else -1


class MockOutputPresenter(OutputPresenter):

    def __init__(self):
        self.valuations = None

    def present(self, i, xn, xb, choice):
        pass

    def present_valuations(self, valuations):
        self.valuations = valuations


class TestPreferenceExperiment(unittest.TestCase):

    def test_mock_run(self):
        acquirer = MockAcquirer()
        presenter = MockInputPresenter()
        output = MockOutputPresenter()
        ex = PreferenceExperiment(
            acquirer,
            presenter,
            output
        )

        # check initial state
        self.assertEqual(
            acquirer.data.preferences(),
            set([(0, 0), (1, 1)])
        )
        self.assertEqual(
            acquirer.data[(0, 0), (1, 1)],
            1
        )

        # run experiment
        ex.run()

        # check end state
        self.assertEqual(
            acquirer.data.preferences(),
            set([(0, 0), (1, 1), (2, 2)])
        )
        self.assertEqual(
            acquirer.data[(0, 0), (2, 2)],
            1
        )

    def test_mock_monitor(self):
        acquirer = MockAcquirer()
        presenter = MockInputPresenter()
        output = MockOutputPresenter()
        ex = PreferenceExperiment(
            acquirer,
            presenter,
            output
        )

        # monitor
        ex.monitor()

        # check valuation update
        self.assertEqual(
            output.valuations,
            acquirer._valuations
        )

    @unittest.skip('long-running')
    def test_run(self):
        # set RNG seed
        np.random.seed(0)
        tf.set_random_seed(0)

        # set up
        bounds = [
            (-3, 6),
            (-3, 6),
        ]
        optimizer = DirectOptimizer(bounds)
        model = BinaryPreferenceModel()
        presenter = MockInputPresenter()
        output = MockOutputPresenter()

        data = UniformPreferenceDict(2)
        a = (0, 0)
        b = (1, 1)
        c = (2, 2)
        d = (3, 3)
        data[a, b] = presenter.get_choice(a, b)
        data[a, c] = presenter.get_choice(a, c)
        data[b, c] = presenter.get_choice(b, c)
        data[b, d] = presenter.get_choice(b, d)
        data[c, d] = presenter.get_choice(c, d)

        # construct experiment
        acquirer = ExpectedImprovementAcquirer(data, model, optimizer)
        ex = PreferenceExperiment(
            acquirer,
            presenter,
            output
        )

        # verify data
        self.assertEqual(len(data), 5)
        self.assertTrue(verify_preferences(data, presenter))

        # run experiment
        ex.run()

        # verify data
        self.assertEqual(len(data), 6)
        self.assertTrue(verify_preferences(data, presenter))

        # run experiment
        ex.run()

        # verify data
        self.assertEqual(len(data), 7)
        self.assertTrue(verify_preferences(data, presenter))


if __name__ == '__main__':
    unittest.main()
