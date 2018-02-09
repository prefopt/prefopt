"""
Output presenter.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from prefopt.experiment import OutputPresenter

__all__ = [
    'StdoutPresenter'
]


class StdoutPresenter(OutputPresenter):
    """Present to stdout."""

    FMT = 'iteration {i}: {xn} vs {xb} -> {choice}'

    def present(self, i, xn, xb, choice):
        print(self.FMT.format(i=i, xn=xn, xb=xb, choice=choice))

    def present_valuations(self, valuations):
        pass
