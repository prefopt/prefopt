"""
Gaussian-process model for binary preferences.
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

all = ['BinaryPreferenceModel']


def compute_latent(f, y, sigma):
    z = tf.stack([f[r] - f[c] for r, c in y.keys()])
    z /= np.sqrt(2) * sigma
    return z


def compute_probit(z):
    dist = tf.distributions.Normal(loc=0., scale=1.)
    return dist.cdf(z)


def compute_logit(z):
    return 1 / (1 + tf.exp(-z))


def encode_observations(y):
    return np.array([(1 if x == 1 else 0) for x in y.values()])


def define_prior(N, D, eps=1.0):
    X = tf.placeholder(tf.float32, [N, D])
    K = rbf(X) + np.eye(N) * eps
    f = MultivariateNormalTriL(
        loc=tf.zeros(N),
        scale_tril=tf.cholesky(K)
    )
    return X, K, f


def define_likelihood(f, y, sigma, compute_link):
    z = compute_latent(f, y, sigma)
    phi = compute_link(z)
    d = Bernoulli(probs=phi)
    return d


def define_variational_distribution(N):
    qf = Normal(
        loc=tf.Variable(tf.random_normal([N])),
        scale=tf.nn.softplus(tf.Variable(tf.random_normal([N])))
    )
    return qf


def define_posterior_predictive(D, X, K, f_map):
    x = tf.placeholder(tf.float32, [None, D])
    k = rbf(X, x)
    K_inv = tf.matrix_inverse(K)
    mu = tf.reduce_sum(
        tf.matmul(tf.transpose(k), K_inv) * f_map,
        axis=1
    )
    var = 1.0 - tf.reduce_sum(
        tf.matmul(tf.transpose(k), K_inv) * tf.transpose(k),
        axis=1
    )
    return x, mu, var


class BinaryPreferenceModel(PreferenceModel):
    """
    Gaussian-process model for binary preferences.

    Depending on the choice of the link function, the model is either a
    Thurstone-Mosteller (probit) or a Bradley-Terry (logit) model.

    Attributes
    ----------
    n_iter : int, optional (default: 500)
        Number of optimization iterations.
    n_samples : int, optional (default: 1)
        Number of samples for calculating stochastic gradient.
    sigma : float, optional (default: 1.0)
        Standard deviation of observation noise.
    link : str, optional (default: probit)
        Link function. Can either be probit in which case the model is a
        Thurstone-Mosteller model or logit in which case the model is a
        Bradley-Terry model.
    """

    def __init__(self, n_iter=500, n_samples=1, sigma=1.0, link='probit'):
        self.sigma = sigma
        self.n_iter = n_iter
        self.n_samples = n_samples

        if link == 'probit':
            self.compute_link = compute_probit
        elif link == 'logit':
            self.compute_link = compute_logit
        else:
            raise ValueError('Invalid link function: {}'.format(link))

    def fit(self, X, y):
        # make copy of data
        self.X = X.copy()
        self.y = y.copy()
        N, D = X.shape

        # preprocess data
        d = encode_observations(self.y)

        # define prior
        self.X_, K, f = define_prior(N, D)

        # define likelihood
        d_ = define_likelihood(f, self.y, self.sigma, self.compute_link)

        # define variational distribution
        qf = define_variational_distribution(N)

        # perform inference
        inference = KLqp({f: qf}, data={self.X_: self.X, d_: d})
        inference.run(n_iter=self.n_iter, n_samples=self.n_samples)

        # define posterior predictive mean and variance
        f_map = qf.mean().eval()
        self.x_, self.mu, self.var = define_posterior_predictive(
            D, self.X_, K, f_map)

    def mean(self, X):
        """The posterior mean function evaluated at points X."""
        return self.mu.eval(feed_dict={self.x_: X, self.X_: self.X})

    def variance(self, X):
        """The posterior variance function evaluated at points X."""
        return self.var.eval(feed_dict={self.x_: X, self.X_: self.X})
