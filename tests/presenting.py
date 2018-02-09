"""
Low-level specifications for presenting steps.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from prefopt.presenter import FunctionInputPresenter

# pylint: disable=no-self-use


def linear_function(x):
    return x


class GivenAFunctionInputPresenter(object):

    def should_return_one_if_first_item_is_preferred(self):
        # given
        better, worse = 1, 0
        presenter = FunctionInputPresenter(linear_function)

        # when
        choice = presenter.get_choice(better, worse)

        # then
        assert choice == 1

    def should_return_minus_one_if_second_item_is_preferred(self):
        # given
        better, worse = 1, 0
        presenter = FunctionInputPresenter(linear_function)

        # when
        choice = presenter.get_choice(worse, better)

        # then
        assert choice == -1
