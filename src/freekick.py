from .board import *


def freekick_action(board: Board, goal_threshold=0.4):
    ball_vector = Vector.from_point(
        board.ball.position - board.controlled_player.position
    )
    goal_vector = Vector.from_point(board.opponent_goal_position - board.ball.position)
    angle = abs(goal_vector.angle(grade=True))

    if ball_vector.length() < 0.1:
        if goal_vector.length() < goal_threshold and angle < 45:
            return board.set_action(Action.Shot, goal_vector, power=4)
        elif angle > 60:
            return board.set_action(Action.HighPass, goal_vector, power=4)
        else:
            player_positions = [
                x.position
                for x in board.my_team.values()
                if x.role != PlayerRole.GoalKeeper and x != board.controlled_player
            ]
            target = sorted(player_positions, key=lambda x: x.x)[0]
            vector = Vector.from_point(target - board.controlled_player.position)
            return board.set_action(Action.ShortPass, vector, power=1)
    else:
        return board.set_action(None, vector=ball_vector, release_direction=True)
