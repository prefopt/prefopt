"""
Tests for prefopt.experiment.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

from prefopt.acquisition import Acquirer
from prefopt.data import UniformPreferenceDict
from prefopt.experiment import (
    InputPresenter,
    OutputPresenter,
    PreferenceExperiment
)


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


if __name__ == '__main__':
    unittest.main()
