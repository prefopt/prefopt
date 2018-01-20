"""
Output presenter.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from prefopt.experiment import OutputPresenter


class StdoutPresenter(OutputPresenter):
    """Present to stdout."""

    FMT = 'iteration {i}: {xn} vs {xb} -> {choice}'

    def present(self, i, xn, xb, choice):
        print(self.FMT.format(i, xn, xb, choice))
