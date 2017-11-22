"""
Tests for prefopt.optimization.direct.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import numpy as np

from prefopt.optimization.direct import DirectOptimizer


class TestOptimization(unittest.TestCase):

    def test_one_dimensional(self):
        bounds = [(-9, 10)]
        optimizer = DirectOptimizer(bounds)

        # linear
        x = optimizer.maximize(lambda x: x)
        self.assertTrue(np.isclose(x, 10))

        # linear
        x = optimizer.maximize(lambda x: -x)
        self.assertTrue(np.isclose(x, -9))

        # quadratic
        x = optimizer.maximize(lambda x: pow(x, 2)),
        self.assertTrue(np.isclose(x, 10))

        # quadratic
        x = optimizer.maximize(lambda x: -pow(x, 2)),
        self.assertTrue(np.isclose(x, 0), x)


if __name__ == '__main__':
    unittest.main()
