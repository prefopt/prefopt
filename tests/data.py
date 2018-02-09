"""
Low-level specifications for preference data.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

from prefopt.data import UniformPreferenceDict


def should_store_and_retrieve_preference_in_one_dimension():
    # given
    a = (0,)
    b = (1,)
    preferences = UniformPreferenceDict(len(a))

    # when
    preferences[a, b] = 1

    # then
    assert preferences[a, b] == 1
    assert preferences[b, a] == -1


def should_store_and_retrieve_preference_in_two_dimensions():
    # given
    a = (0, 1)
    b = (2, 3)
    preferences = UniformPreferenceDict(len(a))

    # when
    preferences[a, b] = 1

    # then
    assert preferences[a, b] == 1
    assert preferences[b, a] == -1


def should_return_correct_number_of_preferences():
    # given
    a = (0,)
    b = (1,)
    c = (2,)
    preferences = UniformPreferenceDict(len(a))

    # when
    preferences[a, b] = 1
    preferences[a, c] = 1

    # then
    assert len(preferences) == 2


def should_return_correct_preference_key_items():
    # given
    a = (0, 1, 2)
    b = (3, 4, 5)
    e = (9, 9, 9)
    f = (1, 1, 1)

    preferences = UniformPreferenceDict(len(a))

    # when
    preferences[a, b] = 1
    preferences[e, f] = -1

    # then
    assert sorted(preferences.preferences()) == [a, f, b, e]


def should_raise_exception_if_keylength_is_invalid():
    # given, when, then
    with pytest.raises(ValueError):
        UniformPreferenceDict('foo')

    # given, when, then
    with pytest.raises(ValueError):
        UniformPreferenceDict(-1)


def should_raise_exception_if_dimension_is_different_than_keylength():
    # given
    a = (0, 1, 2)
    e = (9, 9)
    preferences = UniformPreferenceDict(len(a))

    # when, then
    with pytest.raises(ValueError):
        preferences[a, e] = 1

    # when, then
    with pytest.raises(ValueError):
        preferences[e, a] = 1
