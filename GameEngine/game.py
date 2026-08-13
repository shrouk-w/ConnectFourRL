import numpy as np


class ConnectFour:

    def __init__(self):
        self.rows = 6
        self.columns = 7

        self.board = np.zeros((self.rows, self.columns), dtype=np.int8)

        self.current_player = 1
        self.winner = None
        self.done = False
        self.move_count = 0
        self.last_move = None

    def reset(self):
        self.board = np.zeros((self.rows, self.columns), dtype=np.int8)

        self.current_player = 1
        self.winner = None
        self.done = False
        self.move_count = 0
        self.last_move = None

        return self.get_state()

    def get_state(self):
        return self.board.copy()

    def is_valid_action(self, action):
        if not isinstance(action, (int, np.integer)):
            return False

        if action < 0 or action >= self.columns:
            return False

        if self.board[0][action] != 0:
            return False

        return True

    def get_valid_actions(self):
        valid_actions = []

        for column in range(self.columns):
            if self.is_valid_action(column):
                valid_actions.append(column)

        return valid_actions