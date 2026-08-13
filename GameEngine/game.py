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

    def step(self, action):
        if self.done:
            raise ValueError("Game is already finished.")

        if not self.is_valid_action(action):
            raise ValueError("Invalid action.")

        row = self._find_empty_row(action)

        self.board[row][action] = self.current_player

        self.last_move = (row, action)
        self.move_count += 1

        if self.current_player == 1:
            self.current_player = 2
        else:
            self.current_player = 1

        return self.get_state()

    def _find_empty_row(self, column):
        row = self.rows - 1

        while row >= 0:
            if self.board[row][column] == 0:
                return row

            row -= 1

        raise ValueError("Column is full.")

game = ConnectFour()

game.step(3)
print(game.board)

game.step(3)
print(game.board)

game.step(4)
print(game.board)