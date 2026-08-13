from GameEngine.game import ConnectFour
from GameEngine.gui import Gui
from Agents.random_agent import RandomAgent


game = ConnectFour()
gui = Gui()
random_agent = RandomAgent()


print("Choose game mode:")
print("1 - Human vs Human")
print("2 - Human vs Random Bot")
print("3 - Random Bot vs Random Bot")

mode = input("Mode: ")

game.reset()

while not game.done:
    gui.render(game.get_state())
    print("Current player:", game.current_player)

    valid_actions = game.get_valid_actions()

    # Human vs Human
    if mode == "1":
        action = input("Choose column (0-6): ")

        try:
            action = int(action)
        except ValueError:
            print("Please enter a number.")
            continue

    # Human vs Bot
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
            print("Bot chooses:", action)

    # Bot vs Bot
    elif mode == "3":
        action = random_agent.select_action(valid_actions)
        print("Bot chooses:", action)

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