import numpy as np
from typing import Optional
from kaggle_environments.envs.football.helpers import Action

_c45 = np.sqrt(2) / 2

_DIRECTION_TO_VECTOR = {
    Action.TopLeft: (-_c45, _c45),
    Action.TopRight: (_c45, _c45),
    Action.BottomRight: (_c45, -_c45),
    Action.BottomLeft: (-_c45, -_c45),
    Action.Bottom: (0, -1),
    Action.Top: (0, 1),
    Action.Right: (1, 0),
    Action.Left: (-1, 0),
}


def angle_to_direction(angle: float, grade: bool = True) -> Action:
    if not grade:
        angle = np.degrees(angle)

    abs_angle = abs(angle)
    if abs_angle <= 22.5:
        return Action.Right
    elif abs_angle <= 90 - 22.5:
        return Action.TopRight if angle > 0 else Action.BottomRight
    elif abs_angle <= 90 + 22.5:
        return Action.Top if angle > 0 else Action.Bottom
    elif abs_angle <= 180 - 22.5:
        return Action.TopLeft if angle > 0 else Action.BottomLeft
    else:
        return Action.Left


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"{self.__class__.__name__}(x={round(self.x, 2)}, y={round(self.y, 2)})"

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __add__(self, other):
        return self.__class__(self.x + other.x, self.y + other.y)

    def __rand__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__class__(self.x - other.x, self.y - other.y)

    def __mul__(self, v):
        return self.__class__(self.x * v, self.y * v)

    def __rmul__(self, v):
        return self.__mul__(v)

    def __truediv__(self, v):
        return self.__class__(self.x / v, self.y / v)

    def __hash__(self):
        return hash((self.x, self.y))


class Vector(Point):
    def __repr__(self):
        return f"{self.__class__.__name__}(x={round(self.x, 3)}, y={round(self.y, 3)})"

    @classmethod
    def from_point(cls, p: Point) -> "Vector":
        return Vector(p.x, p.y)

    @classmethod
    def from_polar(
        cls, angle: float, length: float = 1, grade: bool = False
    ) -> "Vector":
        if grade:
            angle *= np.pi / 180
        return Vector(length * np.cos(angle), length * np.sin(angle))

    @classmethod
    def from_direction(cls, direction: Action) -> "Vector":
        return Vector(*_DIRECTION_TO_VECTOR[direction])

    def normalize(self) -> "Vector":
        n = self.length()
        x = self.x / n
        y = self.y / n
        return Vector(x, y)

    def turn(self, a: float, grade: bool = False) -> "Vector":
        if grade:
            a *= np.pi / 180
        cs = np.cos(a)
        sn = np.sin(a)
        x = self.x * cs - self.y * sn
        y = self.x * sn + self.y * cs
        return Vector(x, y)

    def reverse(self) -> "Vector":
        return Vector(-self.x, -self.y)

    def is_empty(self) -> bool:
        return self.x == 0 and self.y == 0

    def angle(self, grade: bool = False) -> float:
        if self.is_empty():
            return np.nan

        if self.x == 0:
            d = self.y * np.inf
        else:
            d = self.y / self.x

        a = np.arctan(d)

        if self.x < 0:
            a = np.pi + a if self.y >= 0 else a - np.pi

        if grade:
            a *= 180 / np.pi

        return a

    def length(self) -> float:
        return np.sqrt(self.x ** 2 + self.y ** 2)

    def to_direction(self) -> Action:
        return angle_to_direction(self.angle(), grade=False)


def euclidean_distance(p1: Point, p2: Point) -> float:
    return Vector.from_point(p1 - p2).length()


def scalar_product(v1: Vector, v2: Vector) -> float:
    return v1.x * v2.x + v1.y * v2.y


def angle_between_vectors(v1: Vector, v2: Vector, grade=False) -> float:
    if v1.is_empty() or v2.is_empty():
        return np.nan

    s = scalar_product(v1.normalize(), v2.normalize())
    if abs(s) > 1:
        s = np.sign(s)

    a = np.arccos(s)
    if grade:
        a *= 180 / np.pi

    return a


class Line:
    def __init__(self, start: Point, end: Point):
        self.start = start
        self.end = end

    def to_vector(self) -> Vector:
        return Vector.from_point(self.end - self.start)

    @property
    def middle(self) -> Point:
        return (self.start + self.end) / 2

    def get_short_direction(
        self,
        point: Point,
        include_start: bool = True,
        include_end: bool = True,
        infinity_line: bool = False,
    ) -> Optional[Vector]:
        if self.start == self.end:
            return Vector.from_point(self.start - point)

        start_to_end = Vector.from_point(self.end - self.start)
        point_to_start = Vector.from_point(self.start - point)

        n = start_to_end.normalize()
        projection = n * scalar_product(n, point_to_start)

        if not infinity_line:
            if scalar_product(projection, n) >= 0:
                if not include_start:
                    return

                return Vector.from_point(self.start - point)

            if projection.length() >= start_to_end.length():
                if not include_end:
                    return

                return Vector.from_point(self.end - point)

        return point_to_start - projection

    def there_is_an_intersection(self, p: Point, v: Vector) -> bool:
        if v.is_empty:
            return False

        start_angle = Vector.from_point(self.start - p).angle()
        end_angle = Vector.from_point(self.end - p).angle()

        return start_angle < v.angle() < end_angle


class Field:
    def __init__(self, x_min: float, x_max: float, y_min: float, y_max: float):
        assert x_min <= x_max and y_min <= y_max

        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

        self.top_right_corner = Point(x_max, y_max)
        self.bottom_right_corner = Point(x_max, y_min)
        self.top_left_corner = Point(x_min, y_max)
        self.bottom_left_corner = Point(x_min, y_min)

        self.top_line = Line(self.top_right_corner, self.top_left_corner)
        self.bottom_line = Line(self.bottom_right_corner, self.bottom_left_corner)
        self.right_line = Line(self.top_right_corner, self.bottom_right_corner)
        self.left_line = Line(self.top_left_corner, self.bottom_left_corner)

    @classmethod
    def from_points(cls, c1: Point, c2: Point) -> "Field":
        x_min, x_max = sorted([c1.x, c2.x])
        y_min, y_max = sorted([c1.y, c2.y])
        return Field(x_min, x_max, y_min, y_max)

    def __contains__(self, p: Point):
        return self.x_max >= p.x >= self.x_min and self.y_max >= p.y >= self.y_min

    @property
    def borders(self):
        return self.top_line, self.bottom_line, self.right_line, self.left_line

    def border_distance(self, p: Point):
        return min(
            line.get_short_direction(p, infinity_line=True).length()
            for line in self.borders
        )
