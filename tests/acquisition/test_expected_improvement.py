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
from prefopt.model import BinaryPreferenceModel
from prefopt.optimization import DirectOptimizer, GridSearchOptimizer

RNG_SEED = 0
CONFIG = {
    'n_iter': 500,
    'n_samples': 1,
    'sigma_signal': 1.0,
    'sigma_noise': 1.0,
    'link': 'logit',
}


@unittest.skip('long-running')
class TestExpectedImprovementAcquirerDirectOptimizer(tf.test.TestCase):

    def test_interface_with_direct_optimizer_scalar_lengthscale(self):
        with self.test_session():
            # set RNG seed
            np.random.seed(RNG_SEED)
            tf.set_random_seed(RNG_SEED)

            # set up
            bounds = [
                (-3, 6),
                (-3, 6),
            ]

            optimizer = DirectOptimizer(bounds)
            model = BinaryPreferenceModel(
                lengthscale=1.0,
                **CONFIG
            )

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
            eps = 0.5
            self.assertTrue(
                (a1 - eps < xn1 < b1) and
                (a2 - eps < xn2 < b2)
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
            acquirer.update(e, b, 1)

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

    def test_interface_with_direct_optimizer_vector_lengthscale(self):
        with self.test_session():
            # set RNG seed
            np.random.seed(RNG_SEED)
            tf.set_random_seed(RNG_SEED)

            # set up
            bounds = [
                (-3, 6),
                (-3, 6),
            ]
            optimizer = DirectOptimizer(bounds)
            lengthscale = np.array([1, 10], np.float32)
            model = BinaryPreferenceModel(
                lengthscale=lengthscale,
                **CONFIG
            )

            data = UniformPreferenceDict(2)
            a = (0, 0)
            b = (1, 0)
            c = (2, 0)
            d = (3, 0)
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
            eps = 0.5
            self.assertTrue(
                (a1 - eps < xn1 < b1)
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

            # test `lengthscale`
            self.assertAllClose(
                model.lengthscale,
                lengthscale
            )

            # test `update`
            e = (-1, 0)
            acquirer.update(e, a, 1)
            acquirer.update(e, b, 1)

            # test `next`
            l1, l2 = (x[0] for x in bounds)
            xn1, xn2 = acquirer.next
            self.assertTrue((l1 < xn1 < a1))

            # test `best`
            best = acquirer.best
            self.assertTrue(np.allclose(e, best), (e, best))

            # test `valuations`
            valuations = acquirer.valuations
            x, f = zip(*valuations)
            self.assertEqual(len(x), len(data.preferences()))
            self.assertTrue(all(a < b for a, b in zip(x, x[1:])))
            self.assertTrue(all(a > b for a, b in zip(f, f[1:])))

    def test_interface_with_direct_optimizer_ard_lengthscale(self):
        with self.test_session():
            # set RNG seed
            np.random.seed(RNG_SEED)
            tf.set_random_seed(RNG_SEED)

            # set up
            bounds = [
                (-3, 6),
                (-3, 6),
            ]
            optimizer = DirectOptimizer(bounds)
            model = BinaryPreferenceModel(
                ard=True,
                **CONFIG
            )

            data = UniformPreferenceDict(2)
            a = (0, 0)
            b = (1, 0)
            c = (2, 0)
            d = (3, 0)
            data[a, b] = 1
            data[a, c] = 1
            data[b, c] = 1
            data[b, d] = 1
            data[c, d] = 1

            ab = (0.5, 0)
            bc = (1.5, 0)
            cd = (2.5, 0)
            data[a, ab] = 1
            data[ab, b] = 1
            data[b, bc] = 1
            data[bc, c] = 1
            data[c, cd] = 1
            data[cd, d] = 1

            data[ab, bc] = 1
            data[ab, cd] = 1
            data[bc, cd] = 1

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
            eps = 1.0
            self.assertTrue(
                (a1 - eps < xn1 < b1),
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

            # test `lengthscale`
            s1, s2 = model.lengthscale
            self.assertTrue(s1 < s2)

            # test `update`
            e = (-1, 0)
            acquirer.update(e, a, 1)
            acquirer.update(e, ab, 1)
            acquirer.update(e, b, 1)
            acquirer.update(e, bc, 1)

            # test `next`
            l1, l2 = (x[0] for x in bounds)
            xn1, xn2 = acquirer.next
            self.assertTrue(
                (l1 <= xn1 < a1),
            )

            # test `best`
            best = acquirer.best
            self.assertTrue(best < b)

            # test `valuations`
            valuations = acquirer.valuations
            x, f = zip(*valuations)
            self.assertEqual(len(x), len(data.preferences()))
            self.assertTrue(all(a < b for a, b in zip(x, x[1:])))
            self.assertTrue(all(a > b for a, b in zip(f, f[1:])))

            # test `lengthscale`
            s1, s2 = model.lengthscale
            self.assertTrue(s1 < s2)


class TestExpectedImprovementAcquirerGridSearch(tf.test.TestCase):

    def test_interface_with_grid_search_optimizer_scalar_lengthscale(self):
        with self.test_session():
            # set RNG seed
            np.random.seed(RNG_SEED)
            tf.set_random_seed(RNG_SEED)

            # set up
            bounds = [
                (-3, 6),
                (-3, 6),
            ]

            optimizer = GridSearchOptimizer(bounds)
            model = BinaryPreferenceModel(
                lengthscale=1.0,
                **CONFIG
            )

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
            eps = 0.5
            self.assertTrue(
                (a1 - eps < xn1 < b1) and
                (a2 - eps < xn2 < b2)
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
            acquirer.update(e, b, 1)

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

    def test_interface_with_grid_search_optimizer_vector_lengthscale(self):
        with self.test_session():
            # set RNG seed
            np.random.seed(RNG_SEED)
            tf.set_random_seed(RNG_SEED)

            # set up
            bounds = [
                (-3, 6),
                (-3, 6),
            ]
            optimizer = GridSearchOptimizer(bounds)
            lengthscale = np.array([1, 10], np.float32)
            model = BinaryPreferenceModel(
                lengthscale=lengthscale,
                **CONFIG
            )

            data = UniformPreferenceDict(2)
            a = (0, 0)
            b = (1, 0)
            c = (2, 0)
            d = (3, 0)
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
            eps = 0.5
            self.assertTrue(
                (a1 - eps < xn1 < b1)
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

            # test `lengthscale`
            self.assertAllClose(
                model.lengthscale,
                lengthscale
            )

            # test `update`
            e = (-1, 0)
            acquirer.update(e, a, 1)
            acquirer.update(e, b, 1)

            # test `next`
            l1, l2 = (x[0] for x in bounds)
            xn1, xn2 = acquirer.next
            self.assertTrue((l1 < xn1 < a1))

            # test `best`
            best = acquirer.best
            self.assertTrue(np.allclose(e, best))

            # test `valuations`
            valuations = acquirer.valuations
            x, f = zip(*valuations)
            self.assertEqual(len(x), len(data.preferences()))
            self.assertTrue(all(a < b for a, b in zip(x, x[1:])))
            self.assertTrue(all(a > b for a, b in zip(f, f[1:])))

    def test_interface_with_grid_search_optimizer_ard_lengthscale(self):
        with self.test_session():
            # set RNG seed
            np.random.seed(RNG_SEED)
            tf.set_random_seed(RNG_SEED)

            # set up
            bounds = [
                (-3, 6),
                (-3, 6),
            ]
            optimizer = GridSearchOptimizer(bounds)
            model = BinaryPreferenceModel(
                ard=True,
                **CONFIG
            )

            data = UniformPreferenceDict(2)
            a = (0, 0)
            b = (1, 0)
            c = (2, 0)
            d = (3, 0)
            data[a, b] = 1
            data[a, c] = 1
            data[b, c] = 1
            data[b, d] = 1
            data[c, d] = 1

            ab = (0.5, 0)
            bc = (1.5, 0)
            cd = (2.5, 0)
            data[a, ab] = 1
            data[ab, b] = 1
            data[b, bc] = 1
            data[bc, c] = 1
            data[c, cd] = 1
            data[cd, d] = 1

            data[ab, bc] = 1
            data[ab, cd] = 1
            data[bc, cd] = 1

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
            eps = 1.0
            self.assertTrue(
                (a1 - eps < xn1 < b1),
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

            # test `lengthscale`
            s1, s2 = model.lengthscale
            self.assertTrue(s1 < s2)

            # test `update`
            e = (-1, 0)
            acquirer.update(e, a, 1)
            acquirer.update(e, ab, 1)
            acquirer.update(e, b, 1)
            acquirer.update(e, bc, 1)

            # test `next`
            l1, l2 = (x[0] for x in bounds)
            xn1, xn2 = acquirer.next
            self.assertTrue(
                (l1 <= xn1 < a1),
            )

            # test `best`
            best = acquirer.best
            self.assertTrue(best < b)

            # test `valuations`
            valuations = acquirer.valuations
            x, f = zip(*valuations)
            self.assertEqual(len(x), len(data.preferences()))
            self.assertTrue(all(a < b for a, b in zip(x, x[1:])))
            self.assertTrue(all(a > b for a, b in zip(f, f[1:])))

            # test `lengthscale`
            s1, s2 = model.lengthscale
            self.assertTrue(s1 < s2)


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

    def test_expected_improvement_1d(self):
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

    def test_expected_improvement_2d(self):
        # zero stddev
        ei = expected_improvement([0] * 2, [0] * 2, 0)
        self.assertTrue(np.allclose([0] * 2, ei))

        # small stddev
        ei = expected_improvement([0] * 2, [1e-10] * 2, 0)
        self.assertTrue(np.allclose([0] * 2, ei))

        # large stddev
        ei = expected_improvement([0] * 2, [10] * 2, 0)
        self.assertTrue(np.allclose([3.9894228040143269] * 2, ei))

        # mean equal to previous best
        ei = expected_improvement([0] * 2, [1] * 2, 0)
        self.assertTrue(np.allclose([0.3989422804014327] * 2, ei))

        # mean greater than previous best
        ei = expected_improvement([1] * 2, [1] * 2, 0)
        self.assertTrue(np.allclose([1.0833154705876864] * 2, ei))

        # mean smaller than previous best
        ei = expected_improvement([0] * 2, [1] * 2, 1)
        self.assertTrue(np.allclose([0.083315470587686291] * 2, ei))

        # negative stddev
        with self.assertRaises(ValueError):
            expected_improvement([0] * 2, [-1] * 2, 0)


if __name__ == '__main__':
    tf.test.main()
