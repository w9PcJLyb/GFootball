from .board import *


def penalty_action(board: Board) -> Action:
    goal_vector = Vector.from_point(
        board.ball.position - board.opponent_goal_position
    )
    ball_vector = Vector.from_point(
        board.ball.position - board.controlled_player.position
    )
    if board.ball.position.x > 0:
        return board.set_action(Action.Shot, vector=goal_vector, power=4)
    else:
        return board.set_action(None, ball_vector, sprint=True)
