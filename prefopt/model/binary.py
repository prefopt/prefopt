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
    return tf.sigmoid(z)


def encode_observations(y):
    return np.array([(1 if x == 1 else 0) for x in y.values()])


def define_ard_lengthscale(D, gamma=None, low=0.1, high=3.0):
    lscale_low = np.array([low] * D, dtype=np.float32)
    lscale_high = np.array([high] * D, dtype=np.float32)
    if gamma is None:
        gamma = Normal(loc=tf.zeros(D), scale=tf.ones(D))
        lengthscale = ((lscale_high - lscale_low) * tf.sigmoid(gamma) +
                       lscale_low)
        return gamma, lengthscale
    else:
        lengthscale = ((lscale_high - lscale_low) * tf.sigmoid(gamma) +
                       lscale_low)
        return None, lengthscale.eval()


def define_ard_variational_distribution(D):
    qgamma = Normal(
        loc=tf.Variable(tf.random_normal([D])),
        scale=tf.nn.softplus(tf.Variable(tf.random_normal([D])))
    )
    return qgamma


def define_prior(N, D, sigma_noise, sigma_signal, lengthscale):
    """
    Define a Gaussian process prior.

    Parameters
    ----------
    N : int
        The number of observations.
    D : int
        The number of input dimensions.
    sigma_noise : float
        The noise variance.
    sigma_signal : float
        The signal variance.
    lengthscale : float or array-like
        The lengthscale parameter. Can either be a scalar or vector of size D
        where D is the number of dimensions of the input space.

    Returns
    -------
    X : tf.placeholder, shape (N, D)
        A placeholder for the input data.
    K : tf.Tensor, shape (N, N)
        The covariance matrix.
    f : edward.RandomVariable, shape (N,)
        The Gaussian process prior.
    """
    # define model
    X = tf.placeholder(tf.float32, [N, D])
    K = (rbf(X, variance=sigma_signal, lengthscale=lengthscale) +
         np.eye(N) * sigma_noise)
    f = MultivariateNormalTriL(
        loc=tf.zeros(N),
        scale_tril=tf.cholesky(K)
    )

    # check dimensions
    assert X.shape == (N, D)
    assert K.shape == (N, N)
    assert f.shape == (N,)

    return X, K, f


def define_likelihood(f, y, sigma_noise, compute_link):
    """
    Define likelihood for binary observations.

    Parameters
    ----------
    f : edward.RandomVariable, shape (N,)
        The Gaussian process prior.
    y : dict
        Mapping from indices corresponding to items to preferences.
    sigma_noise : float
        The standard deviation of observation noise.
    compute_link : func
        The link function.

    Returns
    -------
    d : edward.RandomVariable, shape (N,)
        The observations.
    """
    z = compute_latent(f, y, sigma_noise)
    phi = compute_link(z)
    d = Bernoulli(probs=phi)
    assert d.shape == (len(y),)
    return d


def define_variational_distribution(N):
    qf = Normal(
        loc=tf.Variable(tf.random_normal([N])),
        scale=tf.nn.softplus(tf.Variable(tf.random_normal([N])))
    )
    return qf


def define_posterior_predictive(X, K, f, sigma_signal, sigma_noise,
                                lengthscale):
    """
    Define the posterior predictive mean and variance.

    Parameters
    ----------
    X : tf.placeholder, shape (N, D)
        A placeholder for the input data.
    K : tf.Tensor, shape (N, N)
        The covariance matrix.
    f : edward.RandomVariable, shape (N,)
        The Gaussian process prior.
    sigma_signal : float
        The signal variance.
    sigma_noise : float
        The noise variance.

    Returns
    -------
    x : tf.placeholder, shape (None, D)
        A placeholder for the future data.
    mu : tf.Tensor, shape (None,)
        The mean function.
    var : tf.Tensor, shape (None,)
        The variance function.
    lengthscale : float or array-like
        The lengthscale parameter. Can either be a scalar or vector of size D
        where D is the number of dimensions of the input space.
    """
    N, D = X.shape
    x = tf.placeholder(tf.float32, [None, D])
    k = rbf(X, x, variance=sigma_signal, lengthscale=lengthscale)
    K_inv = tf.matrix_inverse(K)

    mu = tf.reduce_sum(
        tf.matmul(tf.transpose(k), K_inv) * f,
        axis=1
    )

    c = sigma_signal + sigma_noise
    var = c - tf.reduce_sum(
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
        The number of optimization iterations.
    n_samples : int, optional (default: 1)
        The number of samples for calculating stochastic gradient.
    sigma_signal : float, optional (default: 1.0)
        The variance of the signal.
    sigma_noise : float, optional (default: 1.0)
        The variance of the observation noise.
    link : str, optional (default: probit)
        The link function. Can either be probit in which case the model is a
        Thurstone-Mosteller model or logit in which case the model is a
        Bradley-Terry model.
    lengthscale : float or np.array, optional (default: 1.0)
        The lengthscale parameter. Can either be a scalar or vector of size D
        where D is the number of dimensions of the input space.
    ard : bool, optional (default: False)
        Use automatic relevance determination (ARD) instead of fixed
        lengthscales. If True, the value of lengthscale will be ignored.
    """

    def __init__(self, n_iter=500, n_samples=1, sigma_signal=1.0,
                 sigma_noise=1.0, link='logit', lengthscale=1.0, ard=False):
        self.n_iter = n_iter
        self.n_samples = n_samples
        self.sigma_signal = sigma_signal
        self.sigma_noise = sigma_noise
        self.lengthscale = lengthscale
        self.ard = ard

        if link == 'probit':
            self.compute_link = compute_probit
        elif link == 'logit':
            self.compute_link = compute_logit
        else:
            raise ValueError('Invalid link function: {}'.format(link))

    def fit(self, X, y):
        """
        Perfom inference in the preference model.

        This will update the mean and variance function of the model.

        Parameters
        ----------
        X : np.array, shape (N, D)
            The preference items.
        y : dict
            Mapping from indices corresponding to items (rows) in X to
            preferences. For instance, {(0, 1): 1} means that X[0] is preferred
            over X[1].
        """
        # get dimensions
        N, D = X.shape

        # make copy of data
        self.X = X.copy()
        self.y = y.copy()

        # preprocess data
        d = encode_observations(self.y)

        # instantiate dict with variational_distributions
        variational_distributions = {}

        # define ard lengthscales and variational distribution
        if self.ard:
            gamma, self.lengthscale = define_ard_lengthscale(D)
            qgamma = define_ard_variational_distribution(D)
            variational_distributions[gamma] = qgamma

        # define prior
        self.X_, K, f = define_prior(
            N,
            D,
            self.sigma_noise,
            self.sigma_signal,
            self.lengthscale
        )

        # define likelihood
        d_ = define_likelihood(
            f,
            self.y,
            self.sigma_noise,
            self.compute_link
        )

        # define variational distribution for prior
        qf = define_variational_distribution(N)
        variational_distributions[f] = qf

        # perform inference
        inference = KLqp(
            variational_distributions,
            data={self.X_: self.X, d_: d}
        )
        inference.run(n_iter=self.n_iter, n_samples=self.n_samples)

        # define posterior predictive mean and variance
        qf_mean = qf.mean().eval()

        if self.ard:
            qgamma_mean = qgamma.mean().eval()
            _, self.lengthscale = define_ard_lengthscale(D, gamma=qgamma_mean)

        self.x_, self.mu, self.var = define_posterior_predictive(
            self.X_,
            K,
            qf_mean,
            self.sigma_signal,
            self.sigma_noise,
            self.lengthscale
        )

    def mean(self, X):
        """The posterior mean function evaluated at points X."""
        return self.mu.eval(feed_dict={self.x_: X, self.X_: self.X})

    def variance(self, X):
        """The posterior variance function evaluated at points X."""
        return self.var.eval(feed_dict={self.x_: X, self.X_: self.X})
