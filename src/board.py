from typing import Dict, List, Union
from collections import Counter
from kaggle_environments.envs.football.helpers import (
    GameMode,
    PlayerRole,
    sticky_index_to_action,
)

from .models import Player, Ball
from .logger import logger
from .geometry import *


DIRECTION_COMMANDS = {
    Action.TopLeft,
    Action.Top,
    Action.TopRight,
    Action.BottomRight,
    Action.Bottom,
    Action.BottomLeft,
    Action.Right,
    Action.Left,
}

INTERACTION_COMMANDS = {
    Action.LongPass,
    Action.HighPass,
    Action.ShortPass,
    Action.Shot,
    Action.Slide,
}

DRIBBLE_COMMANDS = {
    Action.Sprint,
    Action.ReleaseDirection,
    Action.ReleaseSprint,
    Action.Dribble,
    Action.ReleaseDribble,
}

_TARGET = None
_LAST_PLAYER = None
_LAST_BALL_PLAYER = None
_LAST_COMMANDS = []
_FREEZED_DIRECTION_COUNT = 0
_AVAILABLE_DIRECTIONS = DIRECTION_COMMANDS


class Board:
    _y_scale = 1.5

    x_max = 1
    x_min = -1
    y_max = 0.42 * _y_scale
    y_min = -0.42 * _y_scale
    field = Field(x_min, x_max, y_min, y_max)
    my_penalty_area = Field(x_min, x_min + 0.26, -0.25 * _y_scale, 0.25 * _y_scale)
    opponent_penalty_area = Field(x_max - 0.26, x_max, -0.25 * _y_scale, 0.25 * _y_scale)

    my_goal_position = Point(x_min, 0)
    opponent_goal_position = Point(x_max, 0)

    opponent_posts = (
        Point(x_max, -0.073 / 2 * _y_scale),
        Point(x_max, 0.073 / 2 * _y_scale),
    )

    def __init__(self, obs):
        self.step = 3001 - obs["steps_left"]
        self.steps_left = obs["steps_left"]
        self.game_mode = GameMode(obs["game_mode"])
        self.score = tuple(obs["score"])
        self.goals_difference = self.score[0] - self.score[1]

        logger.info(f"Step {self.step}.")
        logger.info(f"Mode={self.game_mode.name}, score={self.score}.")

        self.my_team = self._get_team(obs, "left")
        self.my_gk = self._find_gk(self.my_team.values())
        self.opponent_team = self._get_team(obs, "right")
        self.opponent_gk = self._find_gk(self.opponent_team.values())
        self.ball = self._get_ball(obs)
        self.controlled_player = self.my_team[obs["active"]]
        self.sticky_actions = {
            sticky_index_to_action[nr]
            for nr, action in enumerate(obs["sticky_actions"])
            if action
        }

        logger.info(
            f"Controlled player: {self.controlled_player}, "
            f"sticky_actions: {[x.name for x in self.sticky_actions]}."
        )
        if not self.ball.player:
            logger.info(f"The {self.ball} belongs to nobody.")
        elif self.ball.player.is_opponent:
            logger.info(f"The {self.ball} controlled by opponent {self.ball.player}.")
        else:
            logger.info(f"The {self.ball} controlled by my {self.ball.player}.")

        self.next_action = None

        global _TARGET, _LAST_PLAYER, _LAST_BALL_PLAYER, _LAST_COMMANDS, _AVAILABLE_DIRECTIONS, _FREEZED_DIRECTION_COUNT
        ball_player = self.ball.player
        ball_player = ball_player.id if ball_player else None

        def __switch_strategy():
            if self.controlled_player.id != _LAST_PLAYER:
                return True
            if _LAST_BALL_PLAYER != ball_player:
                if _LAST_BALL_PLAYER is None and not self.ball.player.is_opponent:
                    return False
                return True
            return False

        if __switch_strategy():
            _TARGET = None
            _LAST_PLAYER = self.controlled_player.id
            _LAST_BALL_PLAYER = ball_player
            _LAST_COMMANDS = []
            _FREEZED_DIRECTION_COUNT = 0
            _AVAILABLE_DIRECTIONS = DIRECTION_COMMANDS  # | {Action.ReleaseDirection}

        logger.info(
            f"__ target = {_TARGET}, direction_count = {_FREEZED_DIRECTION_COUNT}, "
            f"available_directions = {_AVAILABLE_DIRECTIONS}."
        )

        self.available_directions = _AVAILABLE_DIRECTIONS
        if _TARGET is not None:
            self.available_directions = {self.__find_target_direction(_TARGET)}

    def _get_team(self, obs, side: str) -> Dict[int, Player]:
        if side not in ("left", "right"):
            raise ValueError(f"Unknown team side '{side}'.")

        is_opponent = side == "right"

        return {
            id: Player(
                id=id,
                position=Point(x=p[0], y=-p[1] * self._y_scale),
                vector=Vector(x=d[0], y=-d[1] * self._y_scale),
                role=PlayerRole(r),
                tired_factor=tf,
                yellow_card=yc,
                is_opponent=is_opponent,
            )
            for id, (p, d, r, tf, active, yc) in enumerate(
                zip(
                    obs[f"{side}_team"],
                    obs[f"{side}_team_direction"],
                    obs[f"{side}_team_roles"],
                    obs[f"{side}_team_tired_factor"],
                    obs[f"{side}_team_active"],
                    obs[f"{side}_team_yellow_card"],
                )
            )
            if active
        }

    def _get_ball(self, obs) -> Ball:
        ball_owned_team = obs["ball_owned_team"]
        ball_owned_player = obs["ball_owned_player"]

        if ball_owned_team == 0:
            player = self.my_team[ball_owned_player]
        elif ball_owned_team == 1:
            player = self.opponent_team[ball_owned_player]
        else:
            player = None

        p = obs["ball"]
        d = obs["ball_direction"]
        return Ball(
            position=Point(x=p[0], y=-p[1] * self._y_scale),
            altitude=p[2],
            vector=Vector(x=d[0], y=-d[1] * self._y_scale),
            vertical_speed=d[2],
            player=player,
        )

    @staticmethod
    def _find_gk(players) -> Player:
        gk = [p for p in players if p.role == PlayerRole.GoalKeeper]
        assert len(gk) == 1
        return gk[0]

    def get_player(self, id: int, opponent: bool):
        if opponent:
            return self.opponent_team[id]
        else:
            return self.my_team[id]

    def is_my_player_control_the_ball(self) -> bool:
        return self.controlled_player == self.ball.player

    def is_opponent_player_control_the_ball(self) -> bool:
        return self.ball.player and self.ball.player.is_opponent

    def is_offside_position(self, p: Point, turns=0) -> bool:
        return p.x > self.offside_line(turns)

    def offside_line(self, turns=0) -> float:
        offside_line = max(
            p.future_position(turns).x
            for p in self.opponent_team.values()
            if p.role != PlayerRole.GoalKeeper
        )
        return max(offside_line, 0, self.ball.position.x)

    def is_out(self, p: Point) -> bool:
        return p not in self.field

    def distance_from_out(self, p: Point) -> float:
        return self.field.border_distance(p)

    def is_my_penalty_area(self, p: Point) -> bool:
        return p in self.my_penalty_area

    def is_opponent_penalty_area(self, p: Point) -> bool:
        return p in self.opponent_penalty_area

    @staticmethod
    def _add_command(action, power):
        global _LAST_COMMANDS
        _LAST_COMMANDS.append((action, power))

    @staticmethod
    def _last_commands(n: int = 10, p: int = 1) -> List[Action]:
        global _LAST_COMMANDS
        i = n + p
        counter = Counter([a for a, _ in _LAST_COMMANDS[-i:]])
        return [x for x, c in counter.items() if c >= p]

    @property
    def command_count(self) -> int:
        global _LAST_COMMANDS
        return len(_LAST_COMMANDS)

    @staticmethod
    def _maybe_freezed_commands():
        global _LAST_COMMANDS
        if _LAST_COMMANDS:
            action, power = _LAST_COMMANDS[-1]
            if action:
                power -= 1
                if power > 0:
                    return action, power
        return None, 1

    def _get_action(self, new_action, new_power):
        old_action, old_power = self._maybe_freezed_commands()
        if old_action:
            return old_action, old_power

        if new_action and new_action in self._last_commands(p=new_power):
            return None, 1

        return new_action, new_power

    def _freeze_direction(self, freeze_time, direction, target=None):
        global _TARGET, _FREEZED_DIRECTION_COUNT, _AVAILABLE_DIRECTIONS
        if _FREEZED_DIRECTION_COUNT > 0:
            _FREEZED_DIRECTION_COUNT -= 1
            if _FREEZED_DIRECTION_COUNT == 0:
                _TARGET = None
                _AVAILABLE_DIRECTIONS = DIRECTION_COMMANDS  # | {Action.ReleaseDirection}
                return direction

            if _TARGET is not None:
                return self.__find_target_direction(_TARGET)
            else:
                v = Vector.from_direction(direction)
                return sorted(
                    list(_AVAILABLE_DIRECTIONS),
                    key=lambda x: abs(angle_between_vectors(v, Vector.from_direction(x))),
                )[0]

        if freeze_time > 0:
            _FREEZED_DIRECTION_COUNT = freeze_time
            _TARGET = target
            _AVAILABLE_DIRECTIONS = {direction}

        return direction

    def __find_target_direction(self, target):
        target = self.get_player(id=target, opponent=False)

        target_vector = Vector.from_point(
            target.position - self.controlled_player.position
        )
        default = target_vector.to_direction()
        return default

    @staticmethod
    def _can_handle_action():
        global _FREEZED_DIRECTION_COUNT, _LAST_COMMANDS
        return _FREEZED_DIRECTION_COUNT == 0 and Action.Shot not in _LAST_COMMANDS[-10:]

    def set_action(
        self,
        action: Optional[Action],
        vector: Union[Vector, Action, Player],
        dribble=False,
        sprint=False,
        release_direction=False,
        power=1,
        freeze_direction=0,
    ):
        assert action in INTERACTION_COMMANDS or action is None

        if action and not self._can_handle_action():
            logger.debug(f"Can't handle action {action}.")
            action = None
            power = 1

        target = None
        if isinstance(vector, Player):
            target = vector.id
            direction = Vector.from_point(
                vector.position - self.controlled_player.position
            ).to_direction()
        elif isinstance(vector, Action):
            direction = vector
        else:
            direction = vector.to_direction()

        action, power = self._get_action(action, power)

        self._add_command(action, power)
        direction = self._freeze_direction(freeze_direction, direction, target)

        logger.debug(
            f"Set action {action}, {vector}, "
            f"dribble={dribble}, sprint={sprint}, release_direction={release_direction}, "
            f"power={power}, freeze_direction={freeze_direction}."
        )

        if action:
            if action == Action.Slide:
                player = self.controlled_player
                ball_distance = euclidean_distance(player.position, self.ball.position)
                opponent_distance = np.nan
                if self.ball.player:
                    opponent_distance = euclidean_distance(
                        player.position, self.ball.player.position
                    )
            return action

        if direction not in self.sticky_actions:
            return direction

        if not dribble:
            if Action.Dribble in self.sticky_actions:
                return Action.ReleaseDribble
        else:
            if Action.Dribble not in self.sticky_actions:
                return Action.Dribble

        if not sprint:
            if Action.Sprint in self.sticky_actions:
                return Action.ReleaseSprint
        else:
            if Action.Sprint not in self.sticky_actions:
                return Action.Sprint

        if (
            release_direction
            and direction not in self.sticky_actions
            and any(d in self.sticky_actions for d in DIRECTION_COMMANDS)
        ):
            return Action.ReleaseDirection

        return direction
