"""
Active preference learning experiment.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import prefopt.acquisition
import prefopt.data
import prefopt.model
import prefopt.optimization


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

    @abc.abstractmethod
    def present_valuations(self, valuations):
        """
        Present the current valuations of the model.

        Parameters
        ----------
        valuations : iterable
            Iterable of (item, valuation) tuples.
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

    def __init__(self, acquirer, input_presenter, output_presenter,
                 seed_data=None):
        self.acquirer = acquirer
        self.input_presenter = input_presenter
        self.output_presenter = output_presenter
        self.iteration = 0
        if seed_data:
            self.acquirer.update(*seed_data)

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

    def monitor(self):
        """Monitor the current valuations of the model."""
        valuations = self.acquirer.valuations
        self.output_presenter.present_valuations(valuations)


def to_classname(name, suffix=''):
    return ''.join([token.capitalize() for token in name.split('-')]) + suffix


def create_acquirer(acquisition_strategy, model, optimizer, bounds):
    data = prefopt.data.UniformPreferenceDict(len(bounds))
    model = prefopt.model.BinaryPreferenceModel(link=model)

    # pylint: disable=abstract-class-instantiated
    optimizer_classname = to_classname(optimizer, 'Optimizer')
    optimizer_class = getattr(prefopt.optimization, optimizer_classname)
    optimizer = optimizer_class(bounds)

    acquirer_classname = to_classname(acquisition_strategy, 'Acquirer')
    acquirer_class = getattr(prefopt.acquisition, acquirer_classname)

    return acquirer_class(data, model, optimizer)
