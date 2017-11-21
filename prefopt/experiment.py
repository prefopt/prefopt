"""
Active preference learning experiment.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc


class InputPresenter(object):
    """
    Presenter interface for the user input of an experiment.
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_choice(self, a, b):
        """
        Present the user with a pair of items and return their choice.

        Parameters
        ----------
        a : tuple
            Query point.
        b : tuple
            Comparison point.

        Returns
        -------
        choice : int
            The user preference for the choice a vs. b. Possible values are:
            1 (a preferred over b); 0 (a is equivalent to b); -1 (b preferred
            over a).
        """


class OutputPresenter(object):
    """
    Presenter interface for the output of an experiment.
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def present(self, i, xn, xb, choice):
        """
        Present the output of a single iteration of the experiment.

        Parameters
        ----------
        i : int
            The current iteration.
        xn : tuple
            The new query point.
        xb : tuple
            The incumbent against which the new query point is compared.
        choice : int
            The user preference for the choice `xn` vs. `xb`.
        """


class PreferenceExperiment(object):
    """
    Active preference learning experiment.

    Parameters
    ----------
    acquirer : PreferenceAcquirer
        Acquirer object that is capable of finding new query points based on
        the optimization of some acquisition function.
    input_presenter : InputPresenter
        Presenter object that is capable of presenting a choice to the user and
        returning their preference.
    output_presenter : OutputPresenter
        Presenter object that is capable of presenting the output of an
        iteration of the experiment.

    Attributes
    ----------
    iteration : int
        The current iteration, starting from zero.
    """

    def __init__(self, acquirer, input_presenter, output_presenter):
        self.acquirer = acquirer
        self.input_presenter = input_presenter
        self.output_presenter = output_presenter
        self.iteration = 0

    def run(self):
        """Run one iteration of the preference experiment."""
        # find next query point by optimizing the acquisition function
        xn = self.acquirer.next

        # get current best point to compare against
        xb = self.acquirer.best

        # record user preference
        choice = self.input_presenter.get_choice(xn, xb)

        # update model and infer valuation function
        self.acquirer.update(xn, xb, choice)

        # present output
        self.output_presenter.present(self.iteration, xn, xb, choice)
        self.iteration += 1
