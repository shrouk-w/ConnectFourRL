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

        if self.done:
            return False

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

        if self._check_win(row, action):
            self.winner = self.current_player
            self.done = True

            return self.get_state()

        if self.move_count == self.rows * self.columns:
            self.done = True

            return self.get_state()

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

    def _check_win(self, row, column):
        player = self.board[row][column]

        directions = [
            (0, 1),  # poziomo
            (1, 0),  # pionowo
            (1, 1),  # przekątna \
            (1, -1)  # przekątna /
        ]

        for row_change, column_change in directions:
            count = 1

            count += self._count_pieces(
                row,
                column,
                row_change,
                column_change,
                player
            )

            count += self._count_pieces(
                row,
                column,
                -row_change,
                -column_change,
                player
            )

            if count >= 4:
                return True

        return False

    def _count_pieces(self, row, column, row_change, column_change, player):
        count = 0

        current_row = row + row_change
        current_column = column + column_change

        while (
                current_row >= 0
                and current_row < self.rows
                and current_column >= 0
                and current_column < self.columns
        ):
            if self.board[current_row][current_column] != player:
                break

            count += 1

            current_row += row_change
            current_column += column_change

        return count

    def get_action_mask(self):
        mask = np.zeros(self.columns, dtype=np.int8)

        for column in range(self.columns):
            if self.is_valid_action(column):
                mask[column] = 1

        return mask

    def is_draw(self):
        if self.done and self.winner is None:
            return True

        return False