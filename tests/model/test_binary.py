"""
Tests for prefopt.model.binary.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
import tensorflow as tf

from prefopt.acquisition.expected_improvement import preprocess_data
from prefopt.data import PreferenceDict
from prefopt.model.binary import (
    BinaryPreferenceModel,
    compute_latent,
    compute_logit,
    compute_probit,
    encode_observations
)


class TestHelperFunctions(tf.test.TestCase):

    def test_compute_latent(self):
        with self.test_session():
            N = 3
            f_ = tf.placeholder(tf.float32, [N])
            f = np.array([2, 1, 0])
            y = collections.OrderedDict([
                ((0, 1), 1),
                ((0, 2), -1),
                ((1, 2), 1),
            ])
            sigma = 2.0
            z = compute_latent(f_, y, sigma)
            w = np.array([1, 2, 1]) / (np.sqrt(2) * sigma)

            self.assertAllClose(
                z.eval(feed_dict={f_: f}),
                w
            )

    def test_compute_probit(self):
        with self.test_session():
            N = 3
            z_ = tf.placeholder(tf.float32, [N])
            z = np.array([0., 1.96, 3.], dtype=np.float32)
            phi = compute_probit(z)
            w = np.array([0.5, 0.9750021, 0.9986501], dtype=np.float32)

            self.assertAllClose(
                phi.eval(feed_dict={z_: z}),
                w
            )

    def test_compute_logit(self):
        with self.test_session():
            N = 3
            z_ = tf.placeholder(tf.float32, [N])
            z = np.array([-5e1, 0, 1e6], dtype=np.float32)
            phi = compute_logit(z)
            w = np.array([0, 0.5, 1], dtype=np.float32)

            self.assertAllClose(
                phi.eval(feed_dict={z_: z}),
                w
            )

    def test_encode_observations(self):
        y = collections.OrderedDict([
            ((0, 1), 1),
            ((0, 2), 1),
            ((1, 2), -1),
        ])
        d = encode_observations(y)
        e = np.array([1, 1, 0])
        self.assertAllEqual(d, e)


class TestModel(tf.test.TestCase):

    def test_invalid_link(self):
        with self.assertRaises(ValueError):
            BinaryPreferenceModel(link='foo')

    def test_fit_simple_probit(self):
        with self.test_session():
            data = PreferenceDict()
            a = (0,)
            b = (1,)
            c = (2,)
            d = (3,)

            data[a, b] = 1
            data[a, c] = 1
            data[c, b] = -1
            data[c, d] = 1

            m = BinaryPreferenceModel(
                n_iter=500,
                n_samples=20,
                sigma=1.,
                link='probit'
            )
            X, y = preprocess_data(data)
            m.fit(X, y)
            mean = m.mean([
                (0,),
                (1,),
                (2,),
                (3,),
            ])
            self.assertTrue(
                all(x > y for x, y in zip(mean, mean[1:]))
            )

    def test_fit_simple_logit(self):
        with self.test_session():
            data = PreferenceDict()
            a = (0,)
            b = (1,)
            c = (2,)
            d = (3,)

            data[a, b] = 1
            data[a, c] = 1
            data[c, b] = -1
            data[c, d] = 1

            m = BinaryPreferenceModel(
                n_iter=500,
                n_samples=20,
                sigma=1.,
                link='logit'
            )
            X, y = preprocess_data(data)
            m.fit(X, y)
            mean = m.mean([
                (0,),
                (1,),
                (2,),
                (3,),
            ])
            self.assertTrue(
                all(x > y for x, y in zip(mean, mean[1:]))
            )


if __name__ == "__main__":
    tf.test.main()
