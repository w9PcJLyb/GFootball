import unittest
import numpy as np

from src.geometry import Vector, scalar_product, angle_between_vectors
from kaggle_environments.envs.football.helpers import Action


class Test(unittest.TestCase):
    """
    python3 -m unittest src/tests/test_vector.py
    """

    def test_angle(self):
        self.assertEqual(Vector(1, 0).angle(), 0)
        self.assertEqual(Vector(0, 1).angle(), np.pi / 2)
        self.assertEqual(Vector(-1, 0).angle(), np.pi)
        self.assertEqual(Vector(0, -1).angle(), -np.pi / 2)
        self.assertEqual(Vector(1, 1).angle(), np.pi / 4)

    def test_distance(self):
        self.assertEqual(Vector(1, 0).length(), 1)
        self.assertEqual(Vector(1, 1).length(), np.sqrt(2))

    def test_turn(self):
        d1 = Vector(1, 1)
        d1 = d1.turn(np.pi / 4)
        self.assertAlmostEqual(d1.x, 0)
        self.assertAlmostEqual(d1.y, np.sqrt(2))

    def test_reverse(self):
        d1 = Vector(1, 1)
        d1 = d1.reverse()
        d2 = Vector(-1, -1)
        self.assertAlmostEqual(d1.x, d2.x)
        self.assertAlmostEqual(d1.y, d2.y)

    def test_action(self):
        self.assertEqual(Vector(1, 1).to_direction(), Action.TopRight)
        self.assertEqual(Vector(1, -1).to_direction(), Action.BottomRight)

    def test_normalize(self):
        d = Vector(1, 2)
        d = d.normalize()
        self.assertAlmostEqual(d.length(), 1)

    def test_scalar_product(self):
        self.assertEqual(scalar_product(Vector(1, 0), Vector(0, 1)), 0)
        self.assertEqual(scalar_product(Vector(1, 0), Vector(1, 0)), 1)
        self.assertEqual(scalar_product(Vector(1, 0), Vector(-1, 0)), -1)

    def test_angle_between_vectors(self):
        self.assertEqual(angle_between_vectors(Vector(1, 0), Vector(1, 0)), 0)
        self.assertEqual(angle_between_vectors(Vector(1, 0), Vector(0, 1)), np.pi / 2)
        self.assertEqual(angle_between_vectors(Vector(-1, 1), Vector(-1, -1)), np.pi / 2)
