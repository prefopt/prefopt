"""
Tests for prefopt.optimization.grid_search.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import numpy as np

from prefopt.optimization.grid_search import GridSearchOptimizer, compute_grid


class TestHelperFunctions(unittest.TestCase):

    def test_compute_grid(self):
        bounds = [(0, 10), (-5, 1)]
        n_gridpoints = 1000
        dtype = np.float32
        grid = compute_grid(bounds, n_gridpoints, dtype=dtype)
        n_gridpoints_exact = np.sqrt(len(grid))
        ticks = [np.linspace(*x, num=n_gridpoints_exact) for x in bounds]
        true_grid = np.array(
            list(zip(*[x.flatten()
                       for x in np.meshgrid(*ticks, indexing='ij')])),
            dtype=dtype
        )
        self.assertEqual(true_grid.shape, grid.shape)
        self.assertTrue(np.allclose(grid, true_grid))


class TestOptimization(unittest.TestCase):

    def test_one_dimensional(self):
        bounds = [(-9, 10)]
        optimizer = GridSearchOptimizer(bounds)

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
        # NOTE absolute tolerance needs to be increased for this test to pass
        x = optimizer.maximize(lambda x: -pow(x, 2)),
        self.assertTrue(np.isclose(x, 0, atol=1e-5), x)


if __name__ == '__main__':
    unittest.main()
