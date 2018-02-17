"""
Input presenter.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from prefopt.experiment import InputPresenter

__all__ = [
    'FunctionInputPresenter',
]


class FunctionInputPresenter(InputPresenter):
    """
    Presenter for utility functions.

    The presenter maps the latent utility function to a preference relation.

    Parameters
    ----------
    func : function
        Utility function to be maximized.
    """

    def __init__(self, func):
        self.func = func

    def get_choice(self, a, b):
        return 1 if self.func(a) >= self.func(b) else -1
