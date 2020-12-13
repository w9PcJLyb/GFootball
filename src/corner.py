from .board import Board
from .geometry import Vector

from kaggle_environments.envs.football.helpers import Action


def corner_action(board: Board) -> Action:
    if board.ball.position.x > 0:
        return board.set_action(
            Action.HighPass,
            vector=Vector.from_point(
                board.opponent_goal_position - board.ball.position
            ),
            power=3,
        )
    else:
        return board.set_action(
            None,
            vector=Vector.from_point(
                board.ball.position - board.controlled_player.position
            ),
            release_direction=True,
        )
