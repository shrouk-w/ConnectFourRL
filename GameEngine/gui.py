class Gui:

    RESET = "\033[0m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    GRAY = "\033[90m"

    def render(self, board):
        print()

        for row in board:
            print("|", end=" ")

            for cell in row:
                symbol = self._get_symbol(cell)
                print(symbol, end=" ")

            print("|")

        print("  0 1 2 3 4 5 6")
        print()

    def _get_symbol(self, cell):
        if cell == 1:
            return self.RED + "X" + self.RESET

        if cell == 2:
            return self.YELLOW + "O" + self.RESET

        return self.GRAY + "." + self.RESET