from pathlib import Path

from GameEngine.game import ConnectFour
from GameEngine.gui import Gui
from Agents.ppo_agent import PpoAgent


checkpoints_directory = Path("checkpoints")

checkpoints = []

for path in checkpoints_directory.iterdir():
    if path.is_dir():
        checkpoints.append(path)

checkpoints.sort()


print("Available models:")
print()

for index in range(len(checkpoints)):
    print(index + 1, "-", checkpoints[index].name)


choice = int(input("Choose model: "))

checkpoint = checkpoints[choice - 1]

print()
print("Loading:", checkpoint.name)

ppo_agent = PpoAgent(checkpoint)

game = ConnectFour()
gui = Gui()

game.reset()

while not game.done:

    gui.render(game.get_state())
    print("Current player:", game.current_player)

    valid_actions = game.get_valid_actions()

    if game.current_player == 1:

        action = ppo_agent.select_action(
            game.get_state(),
            valid_actions,
            game.current_player
        )

        print("PPO chooses:", action)

    else:

        action = input("Choose column (0-6): ")

        try:
            action = int(action)

        except ValueError:
            print("Please enter a number.")
            continue

        if not game.is_valid_action(action):
            print("Invalid move.")
            continue

    game.step(action)


gui.render(game.get_state())

if game.winner is not None:
    print("Player", game.winner, "wins!")

else:
    print("Draw!")