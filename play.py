from GameEngine.game import ConnectFour
from GameEngine.gui import Gui

from Agents.random_agent import RandomAgent
from Agents.heuristic_agent import HeuristicAgent


game = ConnectFour()
gui = Gui()

random_agent = RandomAgent()
heuristic_agent = HeuristicAgent()


print("Choose game mode:")
print("1 - Human vs Human")
print("2 - Human vs Random")
print("3 - Random vs Random")
print("4 - Human vs Heuristic")
print("5 - Random vs Heuristic")
print("6 - Heuristic vs Heuristic")

mode = input("Mode: ")

game.reset()

while not game.done:
    gui.render(game.get_state())
    print("Current player:", game.current_player)

    valid_actions = game.get_valid_actions()

    if mode == "1":
        action = input("Choose column (0-6): ")

        try:
            action = int(action)
        except ValueError:
            print("Please enter a number.")
            continue


    elif mode == "2":
        if game.current_player == 1:
            action = input("Choose column (0-6): ")

            try:
                action = int(action)
            except ValueError:
                print("Please enter a number.")
                continue

        else:
            action = random_agent.select_action(valid_actions)
            print("Random bot:", action)


    elif mode == "3":
        action = random_agent.select_action(valid_actions)
        print("Random bot:", action)


    elif mode == "4":
        if game.current_player == 1:
            action = input("Choose column (0-6): ")

            try:
                action = int(action)
            except ValueError:
                print("Please enter a number.")
                continue

        else:
            action = heuristic_agent.select_action(
                game.get_state(),
                valid_actions,
                game.current_player
            )

            print("Heuristic bot:", action)


    elif mode == "5":
        if game.current_player == 1:
            action = random_agent.select_action(valid_actions)
            print("Random bot:", action)

        else:
            action = heuristic_agent.select_action(
                game.get_state(),
                valid_actions,
                game.current_player
            )

            print("Heuristic bot:", action)


    elif mode == "6":
        action = heuristic_agent.select_action(
            game.get_state(),
            valid_actions,
            game.current_player
        )

        print("Heuristic bot:", action)

    else:
        print("Invalid game mode.")
        break

    if not game.is_valid_action(action):
        print("Invalid move.")
        continue

    game.step(action)


if game.done:
    gui.render(game.get_state())

    if game.winner is not None:
        print("Player", game.winner, "wins!")
    else:
        print("Draw!")