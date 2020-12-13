import unittest
import numpy as np

from src.portion import Interval


class Test(unittest.TestCase):
    """
    python3 -m unittest src/tests/test_portion.py
    """

    def test_init(self):
        i = Interval()
        self.assertEqual(i.borders, [])

        i = Interval(0, 1)
        self.assertEqual(i.borders, [(0, 1)])

        i = Interval((0, 1), (0.5, 2))
        self.assertEqual(i.borders, [(0, 2)])

        i = Interval((0, 1), (2, np.inf))
        self.assertEqual(i.borders, [(0, 1), (2, np.inf)])

    def test_lower(self):
        i = Interval()
        self.assertTrue(np.isnan(i.lower()))

        i = Interval(-np.inf, 0)
        self.assertEqual(i.lower(), -np.inf)

        i = Interval(0, 1)
        self.assertEqual(i.lower(), 0)

        i = Interval((0, 1), (2, 3))
        self.assertEqual(i.lower(), 0)

    def test_upper(self):
        i = Interval()
        self.assertTrue(np.isnan(i.upper()))

        i = Interval(0, np.inf)
        self.assertEqual(i.upper(), np.inf)

        i = Interval(0, 1)
        self.assertEqual(i.upper(), 1)

        i = Interval((0, 1), (2, 3))
        self.assertEqual(i.upper(), 3)

    def test_length(self):
        i = Interval()
        self.assertEqual(i.length(), 0)

        i = Interval(0, 0)
        self.assertEqual(i.length(), 0)

        i = Interval(0, 1)
        self.assertEqual(i.length(), 1)

        i = Interval((0, 1), (2, 3))
        self.assertEqual(i.length(), 2)

        i = Interval(0, np.inf)
        self.assertEqual(i.length(), np.inf)

    def test_get_closest_value(self):
        i = Interval()
        self.assertTrue(np.isnan(i.get_closest_value(0)))

        i = Interval(0, 0)
        self.assertEqual(i.get_closest_value(0), 0)

        i = Interval((0, 1))
        self.assertEqual(i.get_closest_value(0.6), 0.6)

        i = Interval((0, 1), (2, 3))
        self.assertEqual(i.get_closest_value(1.6), 2)

    def test_contains(self):
        i = Interval()
        self.assertTrue(0 not in i)

        i = Interval((0, 1), (2, np.inf))
        self.assertTrue(0 in i)
        self.assertTrue(1.2 not in i)
        self.assertTrue(100 in i)

    def test_and(self):
        self.assertEqual(Interval(0, 1) & Interval(2, 3), Interval())
        self.assertEqual(Interval(0, 1) & Interval(0.5, np.inf), Interval(0.5, 1))

    def test_or(self):
        self.assertEqual(Interval(0, 1) | Interval(2, 3), Interval((0, 1), (2, 3)))
        self.assertEqual(Interval(0, 1) | Interval(0.5, np.inf), Interval(0, np.inf))

    def test_neg(self):
        self.assertEqual(-Interval(0, np.inf), Interval(-np.inf, 0))

    def test_sub(self):
        self.assertEqual(Interval(0, 1) - Interval(0.5, 1), Interval(0, 0.5))
