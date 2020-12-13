from copy import deepcopy
from typing import Optional
from kaggle_environments.envs.football.helpers import Action, PlayerRole

from .board import Board
from .models import Player, Ball, speed_interval
from .logger import logger
from .geometry import Line, Vector, euclidean_distance, angle_between_vectors


def slide_action(board: Board, opponent: Optional[Player] = None) -> Action:
    if board.ball.player is None:
        # nobody controls the ball
        return apply_back_defence(board, opponent)

    ball_position = board.ball.future_position()
    player_position = board.controlled_player.future_position()
    if ball_position.x < player_position.x and board.ball.x < board.controlled_player.x:
        # behind the ball
        return apply_back_defence(board, opponent)

    ball_to_goal = Line(ball_position, board.my_goal_position)
    intercept_vector = ball_to_goal.get_short_direction(player_position)
    if intercept_vector.length() > 0.1:
        # too far for intercept
        return apply_back_defence(board, opponent)

    return apply_front_defence(board)


def apply_front_defence(board: Board) -> Action:
    """
    Stand on the line between the goal and the ball
    """

    player = board.controlled_player
    slide_threshold = player.max_speed + player.body_radius
    ball = board.ball
    player_position = player.future_position()
    ball_position = ball.future_position()

    ball_to_goal = Line(ball_position, board.my_goal_position)
    intercept_vector = ball_to_goal.get_short_direction(player_position)
    player_to_ball = Vector.from_point(ball_position - player_position)

    sprint = True
    if ball_position.x > player.position.x:
        vector = intercept_vector * 2 + player_to_ball
        logger.debug(
            "Slide action: Move to intercept, "
            f"intercept_vector = {intercept_vector}, "
            f"ball_vector = {player_to_ball}."
        )
        if intercept_vector.length() < slide_threshold * 5:
            sprint = False

    else:
        target = ball.future_position(turns=5)
        logger.debug(f"Slide action: trying to catch up, target = {target}.")
        vector = Vector.from_point(target - player.position)

    return board.set_action(None, vector, sprint=sprint, dribble=False)


def apply_back_defence(
    board: Board, opponent: Optional[Player] = None
) -> Optional[Action]:
    """
    Move to intercept.
    """

    player = board.controlled_player

    ball = board.ball
    if not opponent:
        opponent = ball.player

    if opponent.vector.x < 0:
        # trying to predict opponent's next move
        new_vector = __get_opponent_vector(board, opponent)
        opponent = deepcopy(opponent)
        opponent.vector = new_vector

    intercept_interval = speed_interval(opponent.position, opponent.vector, opponent=player)
    if intercept_interval:
        target = opponent.future_position(turns=intercept_interval.lower() + 3)
    else:
        target = opponent.future_position(turns=5)

    vector = Vector.from_point(target - player.position)

    should_slide = __should_slide(board, player, ball, intercept_interval)
    action = Action.Slide if should_slide else None
    logger.debug(
        "Slide action: Move to intercept, "
        f"opponent = {opponent}, "
        f"action = {action}, "
        f"intercept_interval = {intercept_interval}, "
        f"intercept_vector = {vector}."
    )
    return board.set_action(action, vector, sprint=True, dribble=False)


def __get_opponent_vector(board: Board, opponent: Player):
    ball = board.ball

    target = None
    if ball.player != opponent:
        interval = ball.get_intercept_interval(board, opponent, height=opponent.height)
        if interval:
            target = ball.future_position(interval.lower())
        else:
            height_interval = ball.height_interval(height=opponent.height)
            if height_interval:
                target = ball.future_position(height_interval.lower())
            else:
                target = ball.future_position()

        if euclidean_distance(target, opponent.position) < 0.05:
            return opponent.vector

    if not target:
        target = board.my_goal_position

    logger.debug(f"Slide action: Opponent target = {target}.")
    vector = Vector.from_point(target - opponent.position)

    speed = opponent.max_speed * 0.95

    return vector.normalize() * speed


def __should_slide(board: Board, player: Player, ball: Ball, intercept_interval):

    if (
        player.yellow_card
        or not ball.player
        or player.speed < 0.009
        or board.is_my_penalty_area(player.position)
        or board.is_my_penalty_area(player.future_position())
    ):
        return False

    if intercept_interval and intercept_interval.lower() < 20:
        return False

    defence_teammates = __defence_teammates(board)
    ball_distance = euclidean_distance(player.position, ball.position)
    opponent_distance = euclidean_distance(player.position, ball.player.position)

    logger.debug(
        f"Should slide: defence_teammates = {defence_teammates}, ball_distance = {ball_distance}, "
        f"opponent_distance = {opponent_distance}, intercept_interval = {intercept_interval}."
    )

    if defence_teammates and ball_distance < 0.025:
        return True

    if not defence_teammates and opponent_distance < 0.025:
        return True

    return False


def __defence_teammates(board):
    controlled_player = board.controlled_player
    goal_vector = Vector.from_point(
        board.my_goal_position - controlled_player.future_position()
    )
    defence_teammates = []
    for p in board.my_team.values():
        if (
            p == controlled_player
            or p.x > controlled_player.x
            or p.role == PlayerRole.GoalKeeper
        ):
            continue

        teammate_to_controlled = Vector.from_point(
            controlled_player.future_position() - p.future_position()
        )
        angle = abs(angle_between_vectors(goal_vector, teammate_to_controlled))
        if angle < 30:
            defence_teammates.append(p)
    return defence_teammates
