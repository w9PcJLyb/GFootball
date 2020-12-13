from kaggle_environments.envs.football.helpers import PlayerRole

from .board import Board
from .models import Player, field_interval, speed_interval
from .logger import logger
from .portion import Interval
from .geometry import *

SHOT_TH = 0.3


class Target:
    def __init__(self, board: Board, player: Player, target: Player):
        self.board = board
        self.player = player
        self.target = target

        self.__opponents = {}  # turns -> list of dict

        self.vector_to_target = Vector.from_point(
            self.target.future_position(turns=5) - self.player.future_position(turns=5)
        )
        self.diversion_angle = abs(
            angle_between_vectors(self.vector_to_target, player.vector, grade=True)
        )
        self.pass_distance = self.vector_to_target.length()
        self.pass_direction = self.vector_to_target.to_direction()
        self.pass_angle = self.vector_to_target.angle(grade=True)
        self.is_offside_position = self.__is_offside_position()
        self.goal_probability = self.__get_goal_probability(turns=5)
        self.intercept_time, self.field_time = self.__get_intercept_time()

        self.score = self.__get_score()

    def __repr__(self):
        return (
            f"Target({self.player.role} {self.player.id} -> {self.target.role} {self.target.id}, "
            f"pass_direction={self.pass_direction}, pass_distance={round(self.pass_distance, 2)}, "
            f"goal_probability={round(self.goal_probability, 2)}, score = {round(self.score, 2)})"
        )

    def __is_out(self):
        if (
            self.board.is_out(self.target.future_position(0))
            or self.board.distance_from_out(self.target.future_position(0)) < 0.05
        ):
            return True

        if (
            self.board.is_out(self.target.future_position(5))
            or self.board.distance_from_out(self.target.future_position(5)) < 0.05
        ):
            return True

        return False

    def __get_intercept_time(self):
        new_vector = (
            Vector.from_point(
                self.board.opponent_goal_position - self.position
            ).normalize()
            * self.target.max_speed
        )

        interval = field_interval(self.position, new_vector, board=self.board)
        field_time = np.inf
        if interval:
            field_time = interval.upper()

        opponent_team = (
            x
            for x in self.board.opponent_team.values()
            if x.role != PlayerRole.GoalKeeper
        )
        interval = speed_interval(self.position, new_vector, opponent=opponent_team)
        intercept_time = np.inf
        if interval:
            intercept_time = interval.lower()

        return intercept_time, field_time

    def __get_score(self):
        player = self.player

        if self.is_offside_position:
            return -1

        score = 1

        if player.role == PlayerRole.GoalKeeper:
            score -= 0.1

        if self.vector.x < 0:
            score -= 0.2

        num_opponents_around = self.num_opponents_around(max_distance=0.05, turns=5)
        score -= num_opponents_around * 0.1

        num_opponents_ahead = self.num_opponents_ahead(turns=5)
        score -= num_opponents_ahead * 0.05

        if self.__is_out():
            score -= 0.05

        if self.field_time < self.intercept_time:
            score += 0.15

        min_opponent_angle = self.min_opponent_angle(turns=5)
        if min_opponent_angle > 60:
            score += 0.15

        if self.target.future_position(turns=5).x > max(
            x["position"].x for x in self.__get_opponents(turns=5, with_gk=False)
        ):
            score += 0.2

        if score > 0 and self.goal_probability > 0:
            score *= 1 + self.goal_probability

        logger.debug(
            f"Get score: player = {self.target}, score = {round(score, 3)}, "
            f"num_opponents_around = {num_opponents_around}, num_opponents_ahead = {num_opponents_ahead}, "
            f"goal_probability = {round(self.goal_probability, 2)}, "
            f"field_time = {np.round(self.field_time)}, intercept_time = {np.round(self.intercept_time)}, "
            f"min_opponent_angle = {round(min_opponent_angle)}."
        )

        return score

    @property
    def position(self):
        return self.target.position

    @property
    def x(self):
        return self.position.x

    @property
    def y(self):
        return self.position.y

    @property
    def vector(self):
        return self.target.vector

    def goal_vector(self, turns=5):
        return Vector.from_point(
            self.board.opponent_goal_position - self.target.future_position(turns=turns)
        )

    @property
    def is_gk(self) -> bool:
        return self.target.role == PlayerRole.GoalKeeper

    def __is_offside_position(self) -> bool:
        if self.board.controlled_player == self.target:
            return False

        if self.board.is_offside_position(self.target.position, turns=0):
            return True

        if self.board.is_offside_position(
            self.target.future_position(turns=7), turns=7
        ):
            return True

        return False

    def __get_opponents(self, turns=0, with_gk=False):
        if turns not in self.__opponents:
            target_position = self.target.future_position(turns)
            opponents = []
            for p in self.board.opponent_team.values():
                position = p.future_position(turns)
                vector = Vector.from_point(position - target_position)
                opponents.append(
                    dict(
                        position=position,
                        distance=vector.length(),
                        angle=vector.angle(grade=True),
                        gk=p.role == PlayerRole.GoalKeeper,
                    )
                )
            self.__opponents[turns] = opponents

        opponents = self.__opponents[turns]
        if not with_gk:
            opponents = (x for x in opponents if not x["gk"])

        return opponents

    def min_opponent_angle(self, turns=0):
        gaol_angle = self.goal_vector(turns=turns).angle(grade=True)
        return min(
            abs(x["angle"] - gaol_angle)
            for x in self.__get_opponents(turns, with_gk=False)
        )

    def num_opponents_around(self, max_distance=0.2, turns=0):
        return sum(
            1
            for x in self.__get_opponents(turns, with_gk=False)
            if x["distance"] < max_distance
        )

    def num_opponents_ahead(self, max_angle=60, turns=0):
        goal_vector = self.goal_vector(turns=turns)
        target_distance = goal_vector.length()
        target_angle = (goal_vector - Vector(0.15, 0)).angle(grade=True)
        interval = Interval(-max_angle, max_angle)
        if target_angle not in interval:
            if target_angle > 0:
                interval |= Interval(0, target_angle)
            else:
                interval |= Interval(-target_angle, 0)

        return sum(
            1
            for x in self.__get_opponents(turns=turns, with_gk=False)
            if x["angle"] in interval and x["distance"] < target_distance
        )

    def blocked_directions(self, turns=0, block_distance=0.05):
        d = set()
        for p in self.__get_opponents(turns, with_gk=False):
            if p["distance"] < block_distance:
                d.add(angle_to_direction(p["angle"], grade=True))
        return d

    def blocked_interval(self, turns=0, block_distance=0.05):
        d = Interval()
        for p in self.__get_opponents(turns, with_gk=False):
            angle = p["angle"]
            distance = p["distance"]
            if distance < block_distance:
                angular_half_size = np.arctan(0.012 / distance) * 180 / np.pi
                i = Interval(
                    round(angle - angular_half_size), round(angle + angular_half_size)
                )
                if not i:
                    continue
                if i.upper() > 180:
                    i = Interval((i.lower(), 180), (i.upper() - 360, -180))
                if i.lower() < -180:
                    i = Interval((-180, i.upper()), (i.lower() + 360, 180))
                d |= i
        return d

    def can_high_pass(self, turns=5) -> bool:
        if (
            self.is_offside_position
            or self.is_gk
            or euclidean_distance(self.board.my_goal_position, self.position) < 0.4
        ):
            return False

        if (
            self.diversion_angle < 120
            and 0.4 < self.pass_distance < 1.3
            and euclidean_distance(
                self.player.future_position(turns), self.board.opponent_goal_position
            )
            > 0.4
        ):
            if self.num_opponents_ahead(max_angle=45) < 2 and abs(self.pass_angle) < 45:
                return True

            if self.num_opponents_around(turns=turns, max_distance=0.25) < 1:
                return True

        if (
            self.diversion_angle < 90
            and self.player.x > self.board.x_max - 0.3
            and self.x < self.board.offside_line(turns=5) - 0.07
            and self.goal_vector(turns=turns).length() < 0.2
            and abs(self.player.y) > self.board.y_max - 0.2
        ):
            return True

        return False

    def can_long_pass(self, turns=5):
        if self.is_offside_position or self.is_gk:
            return False

        if self.diversion_angle < 120 and self.x > 0 and 0.2 < self.pass_distance < 0.4:
            if (
                self.vector.x > 0
                and self.num_opponents_ahead(turns=turns, max_angle=45) == 0
                and abs(self.pass_angle) < 45
            ):
                return True

        if (
            self.diversion_angle < 90
            and self.player.x > self.board.x_max - 0.3
            and self.x < self.board.offside_line(turns=5) - 0.05
            and self.goal_vector(turns=turns).length() < 0.2
            and self.board.y_max - 0.2 > abs(self.player.y) > self.board.y_max - 0.4
        ):
            return True

        return False

    def __is_free_line(self, turns=0, max_distance=0.05):
        line = Line(
            self.player.future_position(turns), self.target.future_position(turns)
        )
        for opponent in self.board.opponent_team.values():
            if euclidean_distance(opponent.position, self.player.position) < 0.07:
                continue
            intercept_vector = line.get_short_direction(
                opponent.future_position(turns), include_start=False, include_end=False
            )
            if intercept_vector and intercept_vector.length() < max_distance:
                return False
        return True

    def can_short_pass(self):
        if self.is_offside_position:
            return False

        if self.is_gk and abs(self.vector_to_target.angle(grade=True)) > 135:
            return False

        if (
            self.goal_vector().length() > 0.2
            and self.diversion_angle < 120
            and 0.1 < self.pass_distance < 0.4
            and self.num_opponents_around(max_distance=0.15) == 0
            and self.__is_free_line(max_distance=0.1)
        ):
            return True

        if (
            self.goal_vector().length() <= 0.3
            and 0.05 < self.pass_distance < 0.35
            and self.num_opponents_around(max_distance=0.05) == 0
            and self.__is_free_line(max_distance=0.05)
        ):
            return True

        return False

    def _get_angular_goal_size(self, turns=0):
        board = self.board
        if self.x > board.x_max or self.x < 0:
            return 0

        position = self.target.future_position(turns=turns)

        post_vectors = [Vector.from_point(p - position) for p in board.opponent_posts]
        interval = Interval(*[x.angle(grade=True) for x in post_vectors])

        for p in self.__get_opponents(turns=turns, with_gk=True):
            angle = p["angle"]
            if abs(angle) > 90:
                continue

            distance = p["distance"]
            if distance == 0:
                interval = Interval()
                break

            angular_half_size = np.arctan(0.015 / distance) * 180 / np.pi
            p_interval = Interval(angle - angular_half_size, angle + angular_half_size)
            interval -= p_interval

        return interval.length()

    def __get_goal_probability(self, turns=0):
        angular_goal_size = self._get_angular_goal_size(turns=turns)
        if angular_goal_size < 1:
            return 0

        p = min(1, angular_goal_size / 90)

        if not self.vector.is_empty():
            goal_angle = abs(
                angle_between_vectors(
                    self.goal_vector(turns=turns), self.vector, grade=False
                )
            )
            p *= np.cos(goal_angle / 2)
        else:
            goal_angle = 0

        gk_blocked = self.__gk_blocked_goal(turns=0)

        if self.goal_vector(turns=turns).length() < 0.3:
            if not gk_blocked:
                p *= 3

        logger.debug(
            f"Goal probability: player = {self.target}, probability = {round(p, 2)}, "
            f"angular_goal_size = {round(angular_goal_size)}, "
            f"goal_angle = {round(goal_angle * 180 / np.pi)}, gk_blocked = {gk_blocked}."
        )
        return p

    def __gk_blocked_goal(self, turns=0):
        board = self.board
        if self.x > board.x_max or self.x < 0:
            return True

        position = self.target.future_position(turns=turns)

        post_vectors = [Vector.from_point(p - position) for p in board.opponent_posts]
        interval = Interval(*[x.angle(grade=True) for x in post_vectors])

        for p in self.__get_opponents(turns=turns, with_gk=True):
            if not p["gk"]:
                continue

            angle = p["angle"]
            if angle in interval:
                return True

        return False


def make_pass(board: Board, player: Player, speed=True):
    if board.command_count < 2:
        return

    current = Target(board=board, player=player, target=player)
    current_score = current.score
    blocked_directions = current.blocked_directions(1) | current.blocked_directions(5)
    logger.debug(
        f"Make pass: blocked_interval: 0 = {current.blocked_directions(0)}, 1 | 5 = {blocked_directions}."
    )
    logger.debug(
        f"Make pass: offside line: 0 = {board.offside_line(0)}, 7 = {board.offside_line(7)}."
    )

    if abs(board.x_min - player.x) < 0.2 and player.role != PlayerRole.GoalKeeper:
        logger.debug("Pass targeting: The ball to close to my goal, make HighPass")
        my_goal_distance = euclidean_distance(board.my_goal_position, player.position)
        if my_goal_distance < 0.2:
            vectors = [Vector(1, 1), Vector(1, -1)]
            vector = sorted(
                vectors, key=lambda v: abs(angle_between_vectors(player.vector, v))
            )[0]
        else:
            vector = Vector(1, 0)
            if current.num_opponents_around(max_distance=0.05, turns=0):
                return board.set_action(Action.Shot, vector, power=3, sprint=speed)
        return board.set_action(
            Action.HighPass, vector, power=5, freeze_direction=15, sprint=speed
        )

    targets = []
    for p in board.my_team.values():
        if p != player:
            targets.append(Target(board=board, player=player, target=p))

    pass_targets = []
    action_to_attribute = {
        Action.HighPass: "can_high_pass",
        Action.LongPass: "can_long_pass",
        Action.ShortPass: "can_short_pass",
    }
    for x in targets:
        if x.score < current_score * 1.1 or x.pass_direction in blocked_directions:
            continue

        for action in (Action.HighPass, Action.LongPass, Action.ShortPass):
            if x.__getattribute__(action_to_attribute[action])():
                pass_targets.append((action, x))

    if pass_targets:
        logger.debug(f"Pass targeting: targets: {pass_targets}.")
        action, target = sorted(pass_targets, key=lambda t: -t[1].score)[0]
        if action == Action.HighPass:
            if target.pass_distance > 0.6:
                power = 5
            elif target.pass_distance > 0.4:
                power = 3
            else:
                power = 2
            freeze_direction = 15 if target.pass_distance > 0.4 else 10
        elif action == Action.LongPass:
            power = 2 if player.x < 0.5 else 1
            freeze_direction = 10
        else:
            power = 1
            freeze_direction = 10

        return board.set_action(
            action,
            target.target,
            power=power,
            freeze_direction=freeze_direction,
            sprint=speed,
        )

    a = maybe_field_end_pass(board, targets, speed)
    if a:
        return a


def make_shot(board: Board, player: Player, speed=True):
    if board.command_count < 1:
        return

    current = Target(board=board, player=player, target=player)
    player_position = player.future_position(turns=5)
    goal_vector = current.goal_vector(turns=5)
    goal_distance = goal_vector.length()

    goal_score = current.goal_probability

    logger.debug(
        f"Make shot: position = {player.position}, goal_distance = {round(goal_distance, 2)}, "
        f"goal_score = {round(goal_score, 2)}."
    )

    if goal_score > SHOT_TH or goal_distance < 0.15:
        return board.set_action(Action.Shot, goal_vector, sprint=speed)

    if goal_vector.length() > 0.7:
        return

    opponent_gk = board.opponent_gk

    if abs(goal_vector.angle(grade=True)) > 45:
        return

    gk_vector = Vector.from_point(opponent_gk.position - player_position)
    out_of_line = (
        Line(board.opponent_goal_position, opponent_gk.position)
        .get_short_direction(player_position, infinity_line=True)
        .reverse()
    )

    power = 1
    if abs(board.x_max - board.opponent_gk.x) < 0.05:
        power = 2

    if goal_vector.length() > 0.3:
        if min(goal_vector.length(), gk_vector.length()) > 0.25:
            return

        return board.set_action(
            Action.Shot,
            goal_vector.normalize() + out_of_line.normalize(),
            sprint=speed,
            release_direction=True,
            power=power,
        )

    else:
        if min(goal_vector.length(), gk_vector.length()) > 0.22:
            return

        return board.set_action(
            Action.Shot,
            goal_vector.normalize() + out_of_line.normalize(),
            sprint=speed,
            release_direction=False,
            power=power,
        )


def maybe_field_end_pass(board, targets, speed):
    player = board.controlled_player
    if player.x < board.x_min + 0.2:
        return

    interval = field_interval(player.position, player.vector, board)
    if not interval or interval.upper() > 10:
        return

    if player.x > board.x_max - 0.2:
        goal_vector = Vector.from_point(
            board.opponent_goal_position - player.future_position()
        )
        if goal_vector.length() < 0.2:
            return
        else:
            if goal_vector.y < 0:
                direction = Action.BottomLeft
            else:
                direction = Action.TopLeft
            return board.set_action(
                Action.LongPass, direction, power=1, freeze_direction=10, sprint=speed
            )

    target = None
    for p in targets:
        if p.is_gk or p.is_offside_position:
            continue

        if p.pass_distance < 0.3:
            target = p

    if target:
        vector = Vector(1, 0)
        action = Action.LongPass
        return board.set_action(
            action, vector, power=1, freeze_direction=10, sprint=speed
        )
