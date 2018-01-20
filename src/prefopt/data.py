"""
Preference data.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numbers


class PreferenceDict(collections.MutableMapping):
    """
    Dict for preference data.

    The key is assumed to be a tuple (r, c) where r and c are tuples that
    represent preference items and the value is a preference relation (positive
    if r is preferred over c, negative is c is preferred over r, and zero if
    they are equivalent).

    What makes this the PreferenceDict special is the fact that the value v
    corresponding to a key (r, c) can also be looked up using (c, r) as the
    key; in that case, the preference relation will be negated, i.e., -v will
    be returned.

    >>> d = PreferenceDict()
    >>> a, b = (0, 1, 2), (3, 4, 5)
    >>> d[a, b] = 1
    >>> d[b, a] == -1
    True
    >>> e, f = (9, 9), (1, 1)
    >>> d[e, f] = -1
    >>> d[f, e] == 1
    True
    >>> len(d) == 2
    True
    >>> sorted(d.preferences())
    [(0, 1, 2), (1, 1), (3, 4, 5), (9, 9)]
    """

    def __init__(self, init_dict=None, **kwargs):
        self.__data = collections.OrderedDict()
        if init_dict:
            self.update(init_dict)
        if kwargs:
            self.update(kwargs)

    def __setitem__(self, key, value):
        a, b = key
        if b < a:
            a, b = b, a
            value *= -1
        self.__data[a, b] = value

    def __getitem__(self, key):
        a, b = key
        return self.__data[a, b] if a < b else -self.__data[b, a]

    def __delitem__(self, key):
        a, b = key
        if b < a:
            a, b = b, a
        del self.__data[a, b]

    def __len__(self):
        return len(self.__data)

    def __iter__(self):
        return iter(self.__data)

    def preferences(self):
        """Get set of preference items."""
        return set(x for y in self.__data.keys() for x in y)


class UniformPreferenceDict(PreferenceDict):
    """
    PreferenceDict that enforces uniform key length.

    >>> UniformPreferenceDict("foo")
    Traceback (most recent call last):
        ...
    ValueError: key length needs to be positive integer
    >>> UniformPreferenceDict(-1)
    Traceback (most recent call last):
        ...
    ValueError: key length needs to be positive integer
    >>> d = UniformPreferenceDict(3)
    >>> a, b = (0, 1, 2), (3, 4, 5)
    >>> d[a, b] = 1
    >>> d[b, a] == -1
    True
    >>> e = (9, 9)
    >>> d[a, e] = -1
    Traceback (most recent call last):
        ...
    ValueError: invalid key length
    """

    def __init__(self, key_length, **kwargs):
        if not (isinstance(key_length, numbers.Integral) and (key_length > 0)):
            raise ValueError("key length needs to be positive integer")
        self.key_length = key_length
        super(UniformPreferenceDict, self).__init__(kwargs)

    def __setitem__(self, key, value):
        a, b = key
        if not len(a) == len(b) == self.key_length:
            raise ValueError("invalid key length")
        super(UniformPreferenceDict, self).__setitem__(key, value)
