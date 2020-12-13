from kaggle_environments.envs.football.helpers import PlayerRole

from .board import Board
from .logger import logger
from .models import Player, speed_interval, field_interval
from .geometry import *
from .pass_targeting import make_pass, make_shot


def control_action(board: Board) -> Action:
    vector, speed = _make_move(board)

    shot_action = make_shot(board, player=board.controlled_player, speed=speed)
    if shot_action:
        return shot_action

    pass_action = make_pass(board, player=board.controlled_player, speed=speed)
    if pass_action:
        return pass_action

    return board.set_action(action=None, vector=vector, sprint=speed)


def _make_move(board: Board):
    target_vector = Vector.from_point(
        board.opponent_goal_position
        - board.controlled_player.position
        - Vector(0.05, 0)
    )

    if board.controlled_player.x < 0:
        time_th = 15
    elif board.controlled_player.x < 0.5:
        time_th = 15
    else:
        time_th = 15

    vector, speed = _get_best_move(board, target_vector, time_th)
    return vector, speed


def _get_best_move(board: Board, target_vector: Vector, time_th=15):
    player = board.controlled_player

    goal_position = board.my_goal_position
    my_goal_distance = euclidean_distance(goal_position, player.position)
    opponent_goal_vector = Vector.from_point(
        board.opponent_goal_position - player.position
    )
    if (
        opponent_goal_vector.length() < 0.3
        and abs(angle_between_vectors(opponent_goal_vector, player.vector, grade=True))
        < 45
    ):
        return opponent_goal_vector, False

    my_goal_line = Line(goal_position - Point(0, 0.2), goal_position + Point(0, 0.2))

    movement_to_player = {}
    logger.debug("Find best move:")
    for stick in board.available_directions:
        for speed in (True, False):
            p = player.apply(stick, speed)

            opponent_time = __opponent_time(board, p)
            field_time = __field_time(board, p)
            keeper_time = __keeper_time(board, p)

            if my_goal_distance < 0.4 and my_goal_line.there_is_an_intersection(
                p.position, p.vector
            ):
                # penalty if a player move to our post
                field_time /= 2

            movement_to_player[(stick, speed)] = {
                "player": p,
                "opponent_time": opponent_time,
                "field_time": field_time,
                "keeper_time": keeper_time,
            }
            logger.debug(
                f" -- Direction={stick}, speed={speed}: "
                f"vector={p.vector}, opponent_time={round(opponent_time, 1)}, "
                f"field_time={round(field_time, 1)}, keeper_time={round(keeper_time, 1)}."
            )

    def __get_keys_value(_field, _stick, _speed=True):
        return movement_to_player[(_stick, _speed)][_field]

    def opponent_time(_stick, _speed=True):
        return __get_keys_value("opponent_time", _stick, _speed)

    def field_time(_stick, _speed=True):
        return __get_keys_value("field_time", _stick, _speed)

    def keeper_time(_stick, _speed=True):
        return __get_keys_value("keeper_time", _stick, _speed)

    def control_time(_stick, _speed=True, _with_keeper=False):
        t = min(opponent_time(_stick, _speed), field_time(_stick, _speed))
        if _with_keeper:
            t = min(t, keeper_time(_stick, _speed))
        return t

    all_movements = movement_to_player.keys()
    scores = [min(control_time(*m), time_th) for m in all_movements]
    max_score = max(scores)
    best_movements = [
        (Vector.from_direction(d), speed)
        for (d, speed), score in zip(all_movements, scores)
        if score == max_score
    ]
    if not best_movements:
        raise ValueError(
            f"Can't find movement. {board.available_directions, all_movements, scores, best_movements}"
        )

    best_movement = sorted(
        best_movements, key=lambda x: abs(angle_between_vectors(target_vector, x[0]))
    )[0]

    vector, speed = best_movement
    return vector, speed


def __opponent_time(board, player):
    min_t = np.inf
    for x in board.opponent_team.values():
        if (
            x.role == PlayerRole.GoalKeeper
            or euclidean_distance(player.position, x.position) > 0.5
        ):
            continue

        interval = speed_interval(player.position, player.vector, opponent=x)
        if not interval:
            continue

        t = interval.lower()
        if t > min_t:
            continue

        if x.vector.is_empty():
            t *= 1.25
        else:
            intercept_position = player.future_position(turns=t)
            intercept_vector = Vector.from_point(intercept_position - x.position)
            if intercept_vector.is_empty():
                t = 0
            else:
                angle = abs(
                    angle_between_vectors(intercept_vector, x.vector, grade=False)
                )
                t *= 1.5 - np.cos(angle) / 2

        if t > min_t:
            continue

        min_t = t

    return min_t


def __keeper_time(board, player):
    interval = speed_interval(
        player.position, player.vector, opponent=board.opponent_gk
    )
    if not interval:
        return np.inf
    return interval.lower()


def __field_time(board: Board, player: Player):
    interval = field_interval(player.position, player.vector, board)
    if not interval:
        return np.inf
    return interval.upper()
