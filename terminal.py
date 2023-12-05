import numpy as np
from rlgym.utils.gamestates import GameState
from rlgym.utils.terminal_conditions.common_conditions import NoTouchTimeoutCondition, GoalScoredCondition, TimeoutCondition
from rlgym.utils import TerminalCondition


class RL2TerminalCondition(TerminalCondition):
    def __init__(self, tick_skip=8):
        super().__init__()
        self.goal_scored = GoalScoredCondition()
        self.timeout = TimeoutCondition(
            round(300 * 120 / tick_skip))  # End after 5 minutes (if scores end up equal)
        self.no_touch = NoTouchTimeoutCondition(
            round(30 * 120 / tick_skip))  # end after 30 seconds if no touch

    def reset(self, initial_state: GameState):
        self.goal_scored.reset(initial_state)
        self.timeout.reset(initial_state)
        self.no_touch.reset(initial_state)

    def is_terminal(self, current_state: GameState) -> bool:
        if self.goal_scored.is_terminal(current_state):
            return True
        if self.no_touch.is_terminal(current_state):
            return True

        # If overtime (scores are equal), end game after 5 minutes
        _, _, ticks_left = current_state.inverted_ball.angular_velocity
        if ticks_left < 0 and np.isinf(ticks_left):
            return True
        elif ticks_left > 0 and np.isinf(ticks_left) and self.timeout.is_terminal(current_state):
            return True
        return False
