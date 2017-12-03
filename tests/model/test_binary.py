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
    define_likelihood,
    define_posterior_predictive,
    define_prior,
    encode_observations
)


def compute_rbf(X, Y=None, sigma_signal=None, sigma_noise=None,
                lengthscale=None):
    X = X / lengthscale
    XX = np.einsum('ij,ij->i', X, X)[:, np.newaxis]
    if Y is None or Y is X:
        Y = X
        YY = XX
        delta_xy = np.eye(len(X))
    else:
        Y = Y / lengthscale
        YY = np.einsum('ij,ij->i', Y, Y)
        delta_xy = 0
    XY = np.einsum('ij,kj->ik', X, Y)
    K = (XX + YY.T - 2 * XY) / 2.0
    return sigma_signal * np.exp(-K) + delta_xy * sigma_noise


def compute_posterior_predictive(X, x, f, sigma_signal, sigma_noise,
                                 lengthscale):
    N, D = X.shape
    M, E = x.shape

    assert D == E, (D, E)

    C = compute_rbf(
        X,
        sigma_signal=sigma_signal,
        sigma_noise=sigma_noise,
        lengthscale=lengthscale
    )
    k = compute_rbf(
        X,
        Y=x,
        sigma_signal=sigma_signal,
        sigma_noise=sigma_noise,
        lengthscale=lengthscale
    )
    C_inv = np.linalg.inv(C)
    c = sigma_signal + sigma_noise

    assert C_inv.shape == (N, N)
    assert k.shape == (N, M)

    mu = k.T.dot(C_inv).dot(f)
    var = c - np.diag(k.T.dot(C_inv).dot(k))

    assert mu.shape == (M,)
    assert var.shape == (M,)

    return mu, var


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

    def test_define_prior_scalar_lengthscale(self):
        with self.test_session():
            # set up
            X = np.array([
                [0, 0],
                [1, 0],
                [0, 1],
                [1, 1],
                [2.3, 1.7],
            ])
            N, D = X.shape
            sigma_signal = 1.2
            sigma_noise = 0.1
            lengthscale = 1.2

            # define prior
            X_, K, f = define_prior(
                N,
                D,
                sigma_noise=sigma_noise,
                sigma_signal=sigma_signal,
                lengthscale=lengthscale
            )

            self.assertAllClose(
                K.eval(feed_dict={X_: X}),
                compute_rbf(
                    X,
                    sigma_signal=sigma_signal,
                    sigma_noise=sigma_noise,
                    lengthscale=lengthscale
                )
            )

            self.assertAllClose(
                f.mean().eval(feed_dict={X_: X}),
                np.zeros(N)
            )

    def test_define_prior_vector_lengthscale(self):
        with self.test_session():
            # set up
            X = np.array([
                [0, 0],
                [1, 0],
                [0, 1],
                [1, 1],
                [2.3, 1.7],
            ])
            N, D = X.shape
            sigma_signal = 1.2
            sigma_noise = 0.1
            lengthscale = np.array([1.1, 0.4], dtype=np.float32)

            # define prior
            X_, K, f = define_prior(
                N,
                D,
                sigma_noise=sigma_noise,
                sigma_signal=sigma_signal,
                lengthscale=lengthscale
            )

            self.assertAllClose(
                K.eval(feed_dict={X_: X}),
                compute_rbf(
                    X,
                    sigma_signal=sigma_signal,
                    sigma_noise=sigma_noise,
                    lengthscale=lengthscale
                )
            )

            self.assertAllClose(
                f.mean().eval(feed_dict={X_: X}),
                np.zeros(N)
            )

    def test_define_likelihood(self):
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

            d = define_likelihood(f_, y, sigma, compute_logit)
            z = np.array([1, 2, 1]) / (np.sqrt(2) * sigma)
            phi = 1 / (1 + np.exp(-z))
            self.assertAllClose(
                d.mean().eval(feed_dict={f_: f}),
                phi
            )

    def test_define_posterior_predictive_1d_simple(self):
        with self.test_session():
            N, D = 3, 1
            sigma_signal = 0.7
            sigma_noise = 1.5
            lengthscale = 1.2

            X_ = tf.placeholder(tf.float32, [N, D])
            K_ = tf.placeholder(tf.float32, [N, N])
            f_ = tf.placeholder(tf.float32, [N])

            X = np.arange(N)[:, np.newaxis]
            K = compute_rbf(
                    X,
                    sigma_signal=sigma_signal,
                    sigma_noise=sigma_noise,
                    lengthscale=lengthscale
                )
            f = np.zeros(N)

            x_, mu, var = define_posterior_predictive(
                X_,
                K_,
                f_,
                sigma_signal,
                sigma_noise,
                lengthscale
            )

            base_dict = {
                X_: X,
                K_: K,
                f_: f,
            }

            # single query point
            x = np.array([0.5])[:, np.newaxis]

            mu_true, var_true = compute_posterior_predictive(
                X,
                x=x,
                f=f,
                sigma_signal=sigma_signal,
                sigma_noise=sigma_noise,
                lengthscale=lengthscale
            )

            feed_dict = {x_: x}
            feed_dict.update(base_dict)

            self.assertAllClose(
                var.eval(feed_dict=feed_dict),
                var_true
            )
            self.assertAllClose(
                mu.eval(feed_dict=feed_dict),
                mu_true
            )

            # multiple query points
            x = np.array([0, 1.5, 9])[:, np.newaxis]

            mu_true, var_true = compute_posterior_predictive(
                X,
                x=x,
                f=f,
                sigma_signal=sigma_signal,
                sigma_noise=sigma_noise,
                lengthscale=lengthscale
            )

            feed_dict = {x_: x}
            feed_dict.update(base_dict)

            self.assertAllClose(
                var.eval(feed_dict=feed_dict),
                var_true
            )
            self.assertAllClose(
                mu.eval(feed_dict=feed_dict),
                mu_true
            )

    def test_define_posterior_predictive_2d_scalar_lengthscale(self):
        with self.test_session():
            N, D = 3, 2
            sigma_signal = 3.3
            sigma_noise = 1.5
            lengthscale = 0.8

            X_ = tf.placeholder(tf.float32, [N, D])
            K_ = tf.placeholder(tf.float32, [N, N])
            f_ = tf.placeholder(tf.float32, [N])

            X = np.arange(N * D).reshape(N, D)
            K = compute_rbf(
                    X,
                    sigma_signal=sigma_signal,
                    sigma_noise=sigma_noise,
                    lengthscale=lengthscale
                )
            f = np.zeros(N)

            x_, mu, var = define_posterior_predictive(
                X_,
                K_,
                f_,
                sigma_signal,
                sigma_noise,
                lengthscale
            )

            base_dict = {
                X_: X,
                K_: K,
                f_: f,
            }

            # single query point
            x = np.array([[0.5, 2]])

            mu_true, var_true = compute_posterior_predictive(
                X,
                x=x,
                f=f,
                sigma_signal=sigma_signal,
                sigma_noise=sigma_noise,
                lengthscale=lengthscale
            )

            feed_dict = {x_: x}
            feed_dict.update(base_dict)

            self.assertAllClose(
                var.eval(feed_dict=feed_dict),
                var_true
            )
            self.assertAllClose(
                mu.eval(feed_dict=feed_dict),
                mu_true
            )

            # multiple query points
            x = np.array([
                [0, 1.5],
                [9.2, -3],
                [17, 2]
            ])

            mu_true, var_true = compute_posterior_predictive(
                X,
                x=x,
                f=f,
                sigma_signal=sigma_signal,
                sigma_noise=sigma_noise,
                lengthscale=lengthscale
            )

            feed_dict = {x_: x}
            feed_dict.update(base_dict)

            self.assertAllClose(
                var.eval(feed_dict=feed_dict),
                var_true
            )
            self.assertAllClose(
                mu.eval(feed_dict=feed_dict),
                mu_true
            )

    def test_define_posterior_predictive_2d_vector_lengthscale(self):
        with self.test_session():
            N, D = 3, 2
            sigma_signal = 3.3
            sigma_noise = 1.5
            lengthscale = np.array([0.8, 2.3], np.float32)

            X_ = tf.placeholder(tf.float32, [N, D])
            K_ = tf.placeholder(tf.float32, [N, N])
            f_ = tf.placeholder(tf.float32, [N])

            X = np.arange(N * D).reshape(N, D)
            K = compute_rbf(
                    X,
                    sigma_signal=sigma_signal,
                    sigma_noise=sigma_noise,
                    lengthscale=lengthscale
                )
            f = np.zeros(N)

            x_, mu, var = define_posterior_predictive(
                X_,
                K_,
                f_,
                sigma_signal,
                sigma_noise,
                lengthscale
            )

            base_dict = {
                X_: X,
                K_: K,
                f_: f,
            }

            # single query point
            x = np.array([[0.5, 2]])

            mu_true, var_true = compute_posterior_predictive(
                X,
                x=x,
                f=f,
                sigma_signal=sigma_signal,
                sigma_noise=sigma_noise,
                lengthscale=lengthscale
            )

            feed_dict = {x_: x}
            feed_dict.update(base_dict)

            self.assertAllClose(
                var.eval(feed_dict=feed_dict),
                var_true
            )
            self.assertAllClose(
                mu.eval(feed_dict=feed_dict),
                mu_true
            )

            # multiple query points
            x = np.array([
                [0, 1.5],
                [9.2, -3],
                [17, 2]
            ])

            mu_true, var_true = compute_posterior_predictive(
                X,
                x=x,
                f=f,
                sigma_signal=sigma_signal,
                sigma_noise=sigma_noise,
                lengthscale=lengthscale
            )

            feed_dict = {x_: x}
            feed_dict.update(base_dict)

            self.assertAllClose(
                var.eval(feed_dict=feed_dict),
                var_true
            )
            self.assertAllClose(
                mu.eval(feed_dict=feed_dict),
                mu_true
            )


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
