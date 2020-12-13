from .board import Board
from .geometry import Vector

from kaggle_environments.envs.football.helpers import Action


def goalkick_action(board: Board):
    if board.ball.position.x < 0:
        return board.set_action(Action.ShortPass, vector=Vector(1, 0))
    else:
        return board.set_action(
            None,
            vector=Vector.from_point(
                board.ball.position - board.controlled_player.position
            ),
            release_direction=True,
        )
