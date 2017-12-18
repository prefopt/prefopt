"""
Examples of one-dimensional valuation functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import itertools
import sys

import numpy as np
import tensorflow as tf

from bokeh.io import output_file, show
from bokeh.layouts import column, row
from bokeh.plotting import figure

from prefopt.acquisition.expected_improvement import (
    ExpectedImprovementAcquirer
)
from prefopt.experiment import InputPresenter
from prefopt.data import UniformPreferenceDict
from prefopt.model import BinaryPreferenceModel
from prefopt.optimization import GridSearchOptimizer

RNG_SEED = 1
CONFIG = {
    'n_iter': 500,
    'n_samples': 1,
    'sigma_signal': 1.0,
    'sigma_noise': 0.25,
    'link': 'logit',
}


class ExampleInputPresenter(InputPresenter):
    """
    Presenter for example test functions.

    The presenter maps the latent test function to a preference relation.

    Paramters
    ---------
    func : function
        Test function to be minimized (i.e., lower is better).
    """

    def __init__(self, func):
        self.func = func

    def get_choice(self, a, b):
        a = np.array(a)
        b = np.array(b)
        return 1 if self.func(a) < self.func(b) else -1


def parse_args(argv):
    parser = argparse.ArgumentParser()
    return parser.parse_args(argv)


def run_quadratic():
    # set RNG seed
    np.random.seed(RNG_SEED)
    tf.set_random_seed(RNG_SEED)

    # set up
    bounds = [(-5, 5)]

    func = np.square
    input_presenter = ExampleInputPresenter(func)

    optimizer = GridSearchOptimizer(bounds)
    model = BinaryPreferenceModel(
        lengthscale=1.0,
        **CONFIG
    )

    # initialization
    n_initial_observations = 3
    data = UniformPreferenceDict(1)
    initial_observations = np.random.uniform(
        *bounds[0],
        size=n_initial_observations
    ).tolist()
    for a, b in itertools.combinations(initial_observations, 2):
        data[(a,), (b,)] = input_presenter.get_choice(a, b)

    # construct acquirer
    acquirer = ExpectedImprovementAcquirer(data, model, optimizer)

    output_file('1d_quadratic.html')

    n_gridpoints = 1000
    grid = np.linspace(*bounds[0], num=n_gridpoints)
    x = grid[:, np.newaxis]

    n_iter = 5
    plots = []
    for i in range(n_iter):
        xn = acquirer.next
        xb = acquirer.best
        choice = input_presenter.get_choice(xn, xb)
        acquirer.update(xn, xb, choice)

        p = figure(
            title='learned (iteration {}/{})'.format(i + 1, n_iter),
            x_axis_label='x',
            y_axis_label='y',
            x_range=bounds[0],
        )

        # plot mean +/- standard deviation
        mean = acquirer.model.mean(x)
        stddev = np.sqrt(acquirer.model.variance(x))
        lower = mean - stddev
        upper = mean + stddev
        p.line(
            grid,
            mean,
            line_width=3,
            legend='mean',
        )
        p.patch(
            np.concatenate([grid, grid[::-1]]),
            np.concatenate([lower, upper[::-1]]),
            alpha=0.5,
            line_width=1,
            legend='stddev'
        )

        # plot data
        valuations = acquirer.valuations
        xs, ys = zip(*valuations)
        xs_flat = [e[0] for e in xs]
        p.asterisk(
            xs_flat,
            ys,
            size=20,
            color='black',
            legend='data'
        )

        # plot query and incumbent point
        p.circle(
            xn,
            acquirer.model.mean([xn]),
            size=10,
            color='red',
            legend='query'
        )
        p.circle(
            xb,
            acquirer.model.mean([xb]),
            size=10,
            color='green',
            legend='incumbent'
        )

        # plot true valuation
        q = figure(
            title='true valuation',
            x_axis_label='x',
            y_axis_label='y',
            x_range=bounds[0],
        )
        q.line(
            grid,
            func(grid),
            legend='valuation',
            line_width=3,
        )

        # plot data
        q.asterisk(
            xs_flat,
            func(xs_flat),
            size=20,
            color='black',
            legend='data'
        )

        # plot query and incumbent point
        q.circle(
            xn,
            func(xn),
            size=10,
            color='red',
            legend='query'
        )
        q.circle(
            xb,
            func(xb),
            size=10,
            color='green',
            legend='incumbent'
        )

        plots.append(row(p, q))

    show(column(*plots))


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)  # noqa

    run_quadratic()


if __name__ == '__main__':
    sys.exit(main())
