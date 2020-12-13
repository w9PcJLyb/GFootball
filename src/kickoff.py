from .board import *


def kickoff_action(board: Board) -> Action:
    ball_vector = Vector.from_point(
        board.ball.position - board.controlled_player.position
    )
    if ball_vector.length() < 0.1:
        return board.set_action(Action.ShortPass, Vector(-1, 0))
    else:
        return board.set_action(None, ball_vector, sprint=True)
