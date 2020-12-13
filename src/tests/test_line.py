import unittest

from src.board import Vector, Line, Point


class Test(unittest.TestCase):
    """
    python3 -m unittest src/tests/test_line.py
    """

    def test_zero(self):
        line = Line(Point(0, 0), Point(0, 0))
        point = Point(0, 0)
        self.assertEqual(line.get_short_direction(point), Vector(0, 0))

    def test_one_hot(self):
        line = Line(Point(0, 0), Point(0, 0))
        point = Point(1, 0)
        self.assertEqual(line.get_short_direction(point), Vector(-1, 0))

    def test_to_start(self):
        line = Line(Point(0, 0), Point(1, 0))
        point = Point(-1, 0)
        self.assertEqual(line.get_short_direction(point), Vector(1, 0))

    def test_to_end(self):
        line = Line(Point(0, 0), Point(1, 0))
        point = Point(2, 0)
        self.assertEqual(line.get_short_direction(point), Vector(-1, 0))

    def test_get_short_direction(self):
        line = Line(Point(0, 0), Point(0, 1))
        point = Point(0, 0)
        self.assertEqual(line.get_short_direction(point), Vector(0, 0))

        line = Line(Point(-1, 1), Point(1, 1))
        point = Point(0, 2)
        self.assertEqual(line.get_short_direction(point), Vector(0, -1))

        line = Line(Point(1, 1), Point(-1, 1))
        point = Point(0, 2)
        self.assertEqual(line.get_short_direction(point), Vector(0, -1))
