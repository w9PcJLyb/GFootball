import numpy as np
from kaggle_environments.envs.football.helpers import Action

from .slide import slide_action
from .board import Board
from .logger import logger
from .portion import Interval
from .geometry import Vector, euclidean_distance
from .control import control_action


def without_ball_action(board: Board) -> Action:
    ball = board.ball
    player = board.controlled_player

    my_intercept_interval = ball.get_intercept_interval(
        board, player, height=player.height
    )
    height_interval = ball.height_interval(player.height)

    closed_opponent, opponent_intercept_time, opponent_intercept_interval = __find_opponent_intercept_interval(
        board
    )

    if not my_intercept_interval:
        if np.isfinite(opponent_intercept_time):
            logger.debug(
                f"Without ball action: Opponent {closed_opponent} will be first at the ball."
            )
            return __press_opponent(board, player, closed_opponent)
        else:
            logger.debug("Without ball action: Nobody can reach the ball.")
            if height_interval:
                intercept_position = ball.future_position(height_interval.lower())
            else:
                intercept_position = ball.future_position()
            vector = Vector.from_point(intercept_position - player.future_position())
            return board.set_action(action=None, vector=vector, sprint=True)

    # who will the first?
    if opponent_intercept_time < my_intercept_interval.lower():
        logger.debug(
            f"Without ball action: Opponent {closed_opponent} will be first at the ball."
        )
        return __press_opponent(board, player, closed_opponent)
    elif opponent_intercept_time > my_intercept_interval.upper():
        if ball.vector.x > 0:
            intercept_time = min(
                my_intercept_interval.upper(), opponent_intercept_time - 3
            )
        else:
            intercept_time = my_intercept_interval.lower()
    else:
        if ball.vector.x > 0:
            intercept_time = max(
                my_intercept_interval.lower(), opponent_intercept_time - 3
            )
        else:
            intercept_time = my_intercept_interval.lower()

    logger.debug(
        "Without ball action: "
        f"intercept_time = {round(intercept_time, 1)}, "
        f"my_intercept_interval = {my_intercept_interval}, "
        f"opponent_intercept_interval = {opponent_intercept_interval}."
    )

    intercept_position = ball.future_position(intercept_time)
    vector = Vector.from_point(intercept_position - player.position)
    distance = vector.length()
    speed = True if distance > 0.05 else False
    if (
        intercept_time == opponent_intercept_time
        and intercept_time < 6
        and board.command_count > 2
    ):
        if (
            sum(1 for x in board.my_team.values() if x.x < player.x) < 3
            or euclidean_distance(board.my_goal_position, player.position) < 0.4
        ):
            return board.set_action(action=Action.Shot, vector=vector, sprint=speed)

    if (
        intercept_time < opponent_intercept_time
        and intercept_time < 6
        and board.command_count > 2
    ):
        logger.debug(
            f"Without ball action: Call control_actions, intercept_vector = {vector}."
        )
        return control_action(board)

    return board.set_action(action=None, vector=vector, sprint=speed)


def __find_opponent_intercept_interval(board: Board):
    ball = board.ball

    opponent_intercept_interval = Interval()
    closed_opponent, opponent_intercept_time = None, np.inf
    for p in board.opponent_team.values():
        interval = ball.get_intercept_interval(board, p, height=p.height)
        if interval:
            if interval.lower() < opponent_intercept_time:
                opponent_intercept_time = interval.lower()
                closed_opponent = p
            opponent_intercept_interval |= interval

    return closed_opponent, opponent_intercept_time, opponent_intercept_interval


def __press_opponent(board, player, opponent):
    return slide_action(board, opponent)
