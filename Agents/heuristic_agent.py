import random


class HeuristicAgent:

    def select_action(self, state, valid_actions, player):
        if len(valid_actions) == 0:
            raise ValueError("No valid actions available.")

        # 1. if can win - win
        winning_action = self._find_winning_action(
            state,
            valid_actions,
            player
        )

        if winning_action is not None:
            return winning_action

        # 2. if enemy can win - block
        if player == 1:
            opponent = 2
        else:
            opponent = 1

        blocking_action = self._find_winning_action(
            state,
            valid_actions,
            opponent
        )

        if blocking_action is not None:
            return blocking_action

        # 3. prefer middle
        if 3 in valid_actions:
            return 3

        # 4. random legal move
        return random.choice(valid_actions)


    def _find_winning_action(self, state, valid_actions, player):
        for action in valid_actions:
            test_board = state.copy()

            row = self._find_empty_row(test_board, action)

            if row is None:
                continue

            test_board[row][action] = player

            if self._check_win(test_board, row, action, player):
                return action

        return None


    def _find_empty_row(self, board, column):
        row = len(board) - 1

        while row >= 0:
            if board[row][column] == 0:
                return row

            row -= 1

        return None


    def _check_win(self, board, row, column, player):
        directions = [
            (0, 1),
            (1, 0),
            (1, 1),
            (1, -1)
        ]

        for row_change, column_change in directions:
            count = 1

            count += self._count_pieces(
                board,
                row,
                column,
                row_change,
                column_change,
                player
            )

            count += self._count_pieces(
                board,
                row,
                column,
                -row_change,
                -column_change,
                player
            )

            if count >= 4:
                return True

        return False


    def _count_pieces(
        self,
        board,
        row,
        column,
        row_change,
        column_change,
        player
    ):
        rows = len(board)
        columns = len(board[0])

        count = 0

        current_row = row + row_change
        current_column = column + column_change

        while (
            current_row >= 0
            and current_row < rows
            and current_column >= 0
            and current_column < columns
        ):
            if board[current_row][current_column] != player:
                break

            count += 1

            current_row += row_change
            current_column += column_change

        return count