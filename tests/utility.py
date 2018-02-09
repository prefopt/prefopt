"""
Low-level specifications for utility steps.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import pytest

from prefopt.utility import BoundingBox, NegativeQuadraticUtilityFunction

# pylint: disable=no-self-use


class GivenANegativeQuadraticUtilityFunction(object):

    def should_return_correct_bounds_in_one_dimension(self):
        # given
        bounds = BoundingBox([(-2, 3)])

        # when
        utility_function = NegativeQuadraticUtilityFunction(bounds=bounds)

        # then
        assert utility_function.bounds == bounds

    def should_raise_exception_in_higher_dimensions(self):
        # given
        bounds = BoundingBox([(-2, 3), (3, 4)])

        # when, then
        with pytest.raises(ValueError):
            NegativeQuadraticUtilityFunction(bounds=bounds)

    def should_return_square_inside_of_domain(self):
        # given
        bounds = BoundingBox([(-2, 3)])
        x = 2.5

        # when
        utility_function = NegativeQuadraticUtilityFunction(bounds=bounds)

        # then
        assert utility_function.evaluate(x) == -(x * x)

    def should_raise_exception_outside_of_domain(self):
        # given
        bounds = BoundingBox([(-2, 3)])
        x = 5

        # when
        utility_function = NegativeQuadraticUtilityFunction(bounds=bounds)

        # then
        with pytest.raises(ValueError):
            utility_function.evaluate(x)

    def should_return_correct_value_when_used_as_a_callable(self):
        # given
        bounds = BoundingBox([(-2, 3)])
        x = 2.5

        # when
        utility_function = NegativeQuadraticUtilityFunction(bounds=bounds)

        # then
        assert utility_function.evaluate(x) == utility_function(x)

    def should_return_correct_argmax_on_interval_including_zero(self):
        # given
        bounds = BoundingBox([(-2, 3)])
        utility_function = NegativeQuadraticUtilityFunction(bounds=bounds)

        # when
        argmax = utility_function.argmax

        # then
        assert argmax == 0

    def should_return_correct_argmax_on_negative_interval_excluding_zero(self):
        # given
        bounds = BoundingBox([(-3, -2)])
        utility_function = NegativeQuadraticUtilityFunction(bounds=bounds)

        # when
        argmax = utility_function.argmax

        # then
        assert argmax == -2

    def should_return_correct_argmax_on_positive_interval_excluding_zero(self):
        # given
        bounds = BoundingBox([(2, 7)])
        utility_function = NegativeQuadraticUtilityFunction(bounds=bounds)

        # when
        argmax = utility_function.argmax

        # then
        assert argmax == 2


class GivenABoundingBox(object):

    def should_return_correct_dimension_if_zero_dimensional(self):
        # given
        bounds = []

        # when
        bounding_box = BoundingBox(bounds)

        # then
        assert len(bounding_box) == len(bounds)

    def should_return_correct_dimension_if_one_dimensional(self):
        # given
        bounds = [(-2, 3)]

        # when
        bounding_box = BoundingBox(bounds)

        # then
        assert len(bounding_box) == len(bounds)

    def should_return_correct_dimension_if_two_dimensional(self):
        # given
        bounds = BoundingBox([(-2, 3), (3, 4)])

        # when
        bounding_box = BoundingBox(bounds)

        # then
        assert len(bounding_box) == len(bounds)

    def should_raise_exception_if_bounds_are_invalid_in_one_dimension(self):
        # given
        bounds = [(1, 0)]

        # when, then
        with pytest.raises(ValueError):
            BoundingBox(bounds)

    def should_raise_exception_if_bounds_are_invalid_in_two_dimensions(self):
        # given
        bounds = [(0, 1), (1, 0)]

        # when, then
        with pytest.raises(ValueError):
            BoundingBox(bounds)

    def should_draw_random_sample_from_domain_in_one_dimension(self):
        # given
        random.seed(0)
        lower, upper = -2, 3
        bounds = BoundingBox([(lower, upper)])

        # when
        x = bounds.sample()[0]

        # then
        assert lower <= x <= upper

    def should_draw_random_sample_from_domain_in_two_dimensions(self):
        # given
        random.seed(0)
        l0, u0 = -2, 3
        l1, u1 = 4, 8
        bounds = BoundingBox([(l0, u0), (l1, u1)])

        # when
        x0, x1 = bounds.sample()

        # then
        assert (l0 <= x0 <= u0) and (l1 <= x1 <= u1)
