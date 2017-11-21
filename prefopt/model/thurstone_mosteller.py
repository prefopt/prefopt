"""
Thurstone-Mosteller preference model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.inferences import KLqp
from edward.models import (
    Bernoulli,
    MultivariateNormalTriL,
    Normal
)
from edward.util import rbf

from prefopt.acquisition import PreferenceModel


def compute_latent(f, y, sigma):
    z = tf.stack([f[r] - f[c] for r, c in y.keys()])
    z /= np.sqrt(2) * sigma
    return z


def compute_probit(z):
    dist = tf.distributions.Normal(loc=0., scale=1.)
    return dist.cdf(z)


def encode_observations(y):
    return np.array([(1 if x == 1 else 0) for x in y.values()])


class ThurstoneMostellerGaussianProcessModel(PreferenceModel):
    """
    Thurstone-Mosteller model with Gaussian process prior.

    Attributes
    ----------
    n_iter : int, optional (default: 500)
        Number of optimization iterations.
    n_samples : int, optional (default: 1)
        Number of samples for calculating stochastic gradient.
    sigma : float, optional (default: 1.0)
        Standard deviation of observation noise.
    """

    def __init__(self, n_iter=500, n_samples=1, sigma=1.0):
        self.sigma = sigma
        self.n_iter = n_iter
        self.n_samples = n_samples

    def fit(self, X, y):
        # make copy of data
        self.X = X.copy()
        self.y = y.copy()

        # define prior
        N, D = X.shape
        self.X_ = tf.placeholder(tf.float32, [N, D])
        K = rbf(self.X_) + 1
        f = MultivariateNormalTriL(
            loc=tf.zeros(N),
            scale_tril=tf.cholesky(K)
        )

        # define likelihood
        z = compute_latent(f, self.y, self.sigma)
        phi = compute_probit(z)
        d_ = Bernoulli(probs=phi)
        d = encode_observations(self.y)

        # define variational distribution
        qf = Normal(
            loc=tf.Variable(tf.random_normal([N])),
            scale=tf.nn.softplus(tf.Variable(tf.random_normal([N])))
        )

        # perform inference
        inference = KLqp({f: qf}, data={self.X_: self.X, d_: d})
        inference.run(n_iter=self.n_iter, n_samples=self.n_samples)

        # define posterior predictive mean and variance
        self.x_ = tf.placeholder(tf.float32, [None, D])
        k = rbf(self.X_, self.x_)
        K_inv = tf.matrix_inverse(K)
        f_map = qf.mean().eval()

        self.mu = tf.reduce_sum(
            tf.matmul(tf.transpose(k), K_inv) * f_map,
            axis=1
        )
        self.var = 1.0 - tf.reduce_sum(
            tf.matmul(tf.transpose(k), K_inv) * tf.transpose(k),
            axis=1
        )

    def mean(self, X):
        """The posterior mean function evaluated at points X."""
        return self.mu.eval(feed_dict={self.x_: X, self.X_: self.X})

    def variance(self, X):
        """The posterior variance function evaluated at points X."""
        return self.var.eval(feed_dict={self.x_: X, self.X_: self.X})
