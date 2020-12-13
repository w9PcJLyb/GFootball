from .board import *


def throwin_action(board: Board) -> Action:
    my_goal_vector = Vector.from_point(
        board.my_goal_position - board.ball.position
    )
    ball_vector = Vector.from_point(
        board.ball.position - board.controlled_player.position
    )
    if ball_vector.length() < 0.05:
        return board.set_action(Action.ShortPass, my_goal_vector)
    else:
        return board.set_action(None, ball_vector, sprint=True)
