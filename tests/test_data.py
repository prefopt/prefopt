"""
Tests for prefopt.data.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

from prefopt.data import PreferenceDict, UniformPreferenceDict


# TODO https://bugs.python.org/file31761/transformdict3.patch
class TestPreferenceDict(unittest.TestCase):

    def test_basic(self):
        d = PreferenceDict()
        a, b = (0, 1, 2), (3, 4, 5)
        d[a, b] = 1
        self.assertEqual(d[a, b], 1)
        self.assertEqual(d[b, a], -1)

        e, f = (9, 9), (1, 1)
        d[e, f] = -1
        d[f, e] == 1
        self.assertEqual(d[e, f], -1)
        self.assertEqual(d[f, e], 1)

        self.assertEqual(len(d), 2)

    def test_preferences(self):
        d = PreferenceDict()
        a, b = (0, 1, 2), (3, 4, 5)
        d[a, b] = 1

        e, f = (9, 9), (1, 1)
        d[e, f] = -1

        self.assertEqual(
            sorted(d.preferences()),
            [(0, 1, 2), (1, 1), (3, 4, 5), (9, 9)]
        )


class TestUniformPreferenceDict(unittest.TestCase):

    def test_basic(self):
        with self.assertRaises(ValueError):
            UniformPreferenceDict("foo")

        with self.assertRaises(ValueError):
            UniformPreferenceDict(-1)

        d = UniformPreferenceDict(3)
        a, b = (0, 1, 2), (3, 4, 5)
        d[a, b] = 1
        self.assertEqual(d[a, b], 1)
        self.assertEqual(d[b, a], -1)

        e = (9, 9)
        with self.assertRaises(ValueError):
            d[e, a] = -1

        self.assertEqual(len(d), 1)

    def test_preferences(self):
        d = UniformPreferenceDict(3)
        a, b = (0, 1, 2), (3, 4, 5)
        d[a, b] = 1

        e, f = (9, 9, 9), (1, 1, 1)
        d[e, f] = -1

        self.assertEqual(
            sorted(d.preferences()),
            [(0, 1, 2), (1, 1, 1), (3, 4, 5), (9, 9, 9)]
        )


if __name__ == "__main__":
    unittest.main()
