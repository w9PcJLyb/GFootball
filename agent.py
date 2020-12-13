import logging
from src import *
from src.logger import logger
from kaggle_environments.envs.football.helpers import GameMode, Action

logging.raiseExceptions = False


def find_next_command(b: Board):
    gm = b.game_mode

    if gm != GameMode.Normal:
        if gm == GameMode.GoalKick:
            return goalkick_action(b)
        elif gm == GameMode.Corner:
            return corner_action(b)
        elif gm == GameMode.FreeKick:
            return freekick_action(b)
        elif gm == GameMode.ThrowIn:
            return throwin_action(b)
        elif gm == GameMode.Penalty:
            return penalty_action(b)
        else:
            raise ValueError(f"Unknown game mode '{gm}'.")

    if b.is_my_player_control_the_ball():
        return control_action(b)
    elif b.is_opponent_player_control_the_ball():
        return slide_action(b)
    else:
        return without_ball_action(b)


def agent(obs):
    b = Board(obs["players_raw"][0])
    command = find_next_command(b)
    logger.info(f"Send command: {command.name}.")
    return [command.value]
