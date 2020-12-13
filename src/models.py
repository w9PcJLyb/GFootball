import numpy as np
from copy import deepcopy
from typing import List, Optional, Iterable
from kaggle_environments.envs.football.helpers import Action, PlayerRole

from .logger import logger
from .portion import Interval
from .geometry import Point, Vector

DEFAULT_TURNS_TO_FUTURE = 2


class BoardObj:
    def __init__(self, position: Point, vector: Vector):
        self.position = position
        self.vector = vector

    @property
    def x(self):
        return self.position.x

    @property
    def y(self):
        return self.position.y

    @property
    def direction(self):
        return self.vector.to_direction()

    @property
    def speed(self):
        return self.vector.length()

    def future_position(self, turns: int = DEFAULT_TURNS_TO_FUTURE) -> Point:
        return self.position + self.vector * turns


class Player(BoardObj):
    height = 0.5
    max_speed = 0.015
    body_radius = 0.012
    walk_speed = 0.009

    def __init__(
        self,
        id: int,
        position: Point,
        vector: Vector,
        role: PlayerRole,
        tired_factor: float = 0,
        yellow_card: bool = False,
        is_opponent: bool = False,
    ):
        super().__init__(position, vector)
        self.id = id
        self.role = role
        self.tired_factor = tired_factor
        self.yellow_card = yellow_card
        self.is_opponent = is_opponent

    def __repr__(self):
        return f"{self.role.name} {self.id} at {self.position}->{self.vector}"

    def future_position(
        self, turns: int = DEFAULT_TURNS_TO_FUTURE, max_speed=False
    ) -> Point:
        if max_speed and not self.vector.is_empty():
            self.vector = self.vector * self.max_speed / self.vector.length()
        return self.position + self.vector * turns

    def apply(self, stick: Action, speed: bool = False) -> "Player":
        new_obj = deepcopy(self)

        if stick == Action.ReleaseDirection:
            new_obj.vector = Vector(0, 0)
            return new_obj

        acceleration = 0.006
        max_speed = self.max_speed * 0.95

        player_vector = self.vector
        stick_vector = Vector.from_direction(stick)

        acceleration_vector = stick_vector * acceleration
        new_vector = player_vector + acceleration_vector

        if speed and new_vector.length() > max_speed:
            new_vector = new_vector.normalize() * max_speed

        if not speed:
            new_vector = new_vector.normalize() * self.walk_speed

        new_obj.vector = new_vector
        return new_obj


class Ball(BoardObj):
    gravity = 0.098
    windage = -0.0015

    def __init__(
        self,
        position: Point,
        vector: Vector,
        altitude: float,
        vertical_speed: float,
        player: Optional[Player] = None,
    ):
        super().__init__(position, vector)
        self.altitude = altitude
        self.vertical_speed = vertical_speed
        self.player = player

        self._height_to_interval = {}

    def __repr__(self):
        return f"Ball at Point(x={round(self.x, 2)}, y={round(self.y, 2)}, z={round(self.altitude, 2)})->{self.vector}"

    @property
    def z(self):
        return self.altitude

    def future_altitude(self, turns=DEFAULT_TURNS_TO_FUTURE):
        return (
            self.altitude + self.vertical_speed * turns - self.gravity * turns ** 2 / 2
        )

    def height_interval(self, height):
        if height in self._height_to_interval:
            return self._height_to_interval[height]

        d = self.vertical_speed ** 2 + 2 * self.gravity * (self.altitude - height)
        if d < 0:
            interval = Interval(0, np.inf)
        else:
            t1 = (self.vertical_speed - np.sqrt(d)) / self.gravity
            t2 = (self.vertical_speed + np.sqrt(d)) / self.gravity
            interval = Interval(-np.inf, t1) | Interval(t2, np.inf)
            interval &= Interval(0, np.inf)
        self._height_to_interval[height] = interval
        return interval

    def get_intercept_interval(
        self, board, player: Player, height: Optional[float] = None
    ):
        if height is None:
            height = player.height

        height = self.height_interval(height)

        speed = speed_interval(
            self.position, self.vector, opponent=player, acceleration=self.windage
        )
        field = field_interval(self.position, self.vector, board=board)

        if not player.is_opponent:
            logger.debug(
                f"Intercept intervals: height={height}, speed={speed}, field={field}"
            )

        return height & speed & field


def speed_interval(
    position: Point,
    vector: Vector,
    opponent: [Player, List[Player]],
    acceleration: float = 0,
):
    if not isinstance(opponent, Iterable):
        opponent = [opponent]

    interval = Interval()
    for o in opponent:
        i = __speed_interval(
            position, vector, opponent=o, acceleration=acceleration
        )
        interval |= i

    return interval


def __speed_interval(
    position: Point, vector: Vector, opponent: Player, acceleration: float = 0
):
    if not acceleration:
        return __naive_speed_interval(position, vector, opponent)

    speed = vector.length()
    player_speed = opponent.max_speed
    pb = Vector.from_point(position - opponent.position)

    if speed == 0:
        return Interval(pb.length() / player_speed, np.inf)

    direction = vector / speed
    a = acceleration * direction

    dx, dy = pb.x, pb.y
    vx, vy = vector.x, vector.y
    ax, ay = a.x, a.y

    x = (dx + vx * t + ax * t ** 2 / 2 for t in range(1, 101))
    y = (dy + vy * t + ay * t ** 2 / 2 for t in range(1, 101))
    xy = (x ** 2 + y ** 2 for x, y in zip(x, y))

    borders = []
    last_t = False
    speed_2 = player_speed ** 2
    for v, t in zip(xy, range(1, 101)):
        if v <= speed_2 * t ** 2:
            if last_t:
                borders.append((t - 1, t))
            else:
                borders.append((t, t))
            last_t = True
        else:
            last_t = False

    interval = Interval(*borders)
    if interval and interval.lower() < 1:
        return Interval(pb.length() / player_speed, np.inf)

    return interval


def __naive_speed_interval(position: Point, vector: Vector, opponent: Player):
    speed = vector.length()
    pb = Vector.from_point(position - opponent.position)
    dx, dy = vector.x, vector.y

    player_speed = opponent.max_speed

    # at^2 + 2bt + c <= 0
    a = speed ** 2 - player_speed ** 2
    b = dx * pb.x + dy * pb.y
    c = pb.x ** 2 + pb.y ** 2

    if a == 0:
        if b == 0:
            if c == 0:
                return Interval(0, np.inf)
            else:
                return Interval()
        t = -c / (2 * b)
        if b > 0:
            return Interval(0, t)
        else:
            return Interval(max(t, 0), np.inf)

    d = b ** 2 - a * c
    if d < 0:
        return Interval()

    d = np.sqrt(d)
    t1, t2 = sorted([(-b + d) / a, (-b - d) / a])

    if speed >= player_speed:
        return Interval(0, np.inf) & Interval(t1, t2)
    else:
        return Interval(0, np.inf) & Interval((-np.inf, t1), (t2, np.inf))


def field_interval(position: Point, vector: Vector, board):
    x, y = position.x, position.y
    dx, dy = vector.x, vector.y

    if dx > 0:
        tx = Interval((board.x_min - x) / dx, (board.x_max - x) / dx)
    elif dx < 0:
        tx = Interval((board.x_max - x) / dx, (board.x_min - x) / dx)
    else:
        tx = Interval(-np.inf, np.inf)

    if dy > 0:
        ty = Interval((board.y_min - y) / dy, (board.y_max - y) / dy)
    elif dy < 0:
        ty = Interval((board.y_max - y) / dy, (board.y_min - y) / dy)
    else:
        ty = Interval(-np.inf, np.inf)

    return Interval(0, np.inf) & tx & ty
