"""
Tests for prefopt.acquisition.expected_improvement.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import numpy as np
import tensorflow as tf

from prefopt.acquisition.expected_improvement import (
        ExpectedImprovementAcquirer,
        expected_improvement,
        preprocess_data
)
from prefopt.data import UniformPreferenceDict
from prefopt.model.thurstone_mosteller import (
    ThurstoneMostellerGaussianProcessModel
)
from prefopt.optimization import DirectOptimizer


class TestExpectedImprovementAcquirer(tf.test.TestCase):

    def test_public_interface(self):
        with self.test_session():
            # set RNG seed
            np.random.seed(0)
            tf.set_random_seed(0)

            # set up
            bounds = [
                (-3, 6),
                (-3, 6),
            ]
            optimizer = DirectOptimizer(bounds)
            model = ThurstoneMostellerGaussianProcessModel()

            data = UniformPreferenceDict(2)
            a = (0, 0)
            b = (1, 1)
            c = (2, 2)
            d = (3, 3)
            data[a, b] = 1
            data[a, c] = 1
            data[b, c] = 1
            data[b, d] = 1
            data[c, d] = 1

            # construct acquirer
            acquirer = ExpectedImprovementAcquirer(data, model, optimizer)

            # test that `next` needs to be called before `best`
            with self.assertRaises(ValueError):
                acquirer.best

            # test that `next` needs to be called before `valuations`
            with self.assertRaises(ValueError):
                acquirer.valuations

            # test `next`
            a1, a2 = a
            b1, b2 = b
            xn = xn1, xn2 = acquirer.next
            self.assertTrue(
                (a1 < xn1 < b1) and
                (a2 < xn2 < b2)
            )
            self.assertTrue(np.allclose(xn, acquirer.next))

            # test `best`
            best = acquirer.best
            self.assertTrue(np.allclose(a, best))
            self.assertTrue(np.allclose(best, acquirer.best))

            # test `valuations`
            valuations = acquirer.valuations
            x, f = zip(*valuations)
            self.assertEqual(len(x), len(data.preferences()))
            self.assertTrue(all(a < b for a, b in zip(x, x[1:])))
            self.assertTrue(all(a > b for a, b in zip(f, f[1:])))

            # test `update`
            e = (-1, -1)
            acquirer.update(e, a, 1)

            # test `next`
            l1, l2 = (x[0] for x in bounds)
            xn1, xn2 = acquirer.next
            self.assertTrue(
                (l1 < xn1 < a1) and
                (l2 < xn2 < a2),
            )

            # test `best`
            best = acquirer.best
            self.assertTrue(np.allclose(e, best))

            # test `valuations`
            valuations = acquirer.valuations
            x, f = zip(*valuations)
            self.assertEqual(len(x), len(data.preferences()))
            self.assertTrue(all(a < b for a, b in zip(x, x[1:])))
            self.assertTrue(all(a > b for a, b in zip(f, f[1:])))


class TestExpectedImprovementHelperFunctions(unittest.TestCase):

    def test_preprocess_data(self):
        pd = UniformPreferenceDict(2)
        a = (0, 1)
        b = (2, 3)
        c = (4, 5)
        d = (6, 7)
        e = (8, 9)

        pd[a, b] = 1
        pd[a, c] = 1
        pd[a, d] = 1
        pd[c, b] = -1
        pd[d, e] = 0

        X, y = preprocess_data(pd)
        A = np.array([
             [0, 1],
             [2, 3],
             [4, 5],
             [6, 7],
             [8, 9],
        ])
        self.assertTrue(np.allclose(A, X))

        b = {
            (0, 1): 1,
            (0, 2): 1,
            (0, 3): 1,
            (1, 2): 1,
            (3, 4): 0,
        }
        self.assertEqual(y, b)


class TestExpectedImprovementFunction(unittest.TestCase):

    def test_expected_improvement(self):
        # zero stddev
        ei = expected_improvement(0, 0, 0)
        self.assertAlmostEqual(0, ei)

        # small stddev
        ei = expected_improvement(0, 1e-10, 0)
        self.assertAlmostEqual(0, ei)

        # large stddev
        ei = expected_improvement(0, 10, 0)
        self.assertAlmostEqual(3.9894228040143269, ei)

        # mean equal to previous best
        ei = expected_improvement(0, 1, 0)
        self.assertAlmostEqual(0.3989422804014327, ei)

        # mean greater than previous best
        ei = expected_improvement(1, 1, 0)
        self.assertAlmostEqual(1.0833154705876864, ei)

        # mean smaller than previous best
        ei = expected_improvement(0, 1, 1)
        self.assertAlmostEqual(0.083315470587686291, ei)

        # negative stddev
        with self.assertRaises(ValueError):
            expected_improvement(0, -1, 0)


if __name__ == '__main__':
    tf.test.main()
