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

game = ConnectFour()

print(game.board)
print("Current player:", game.current_player)
print("Done:", game.done)