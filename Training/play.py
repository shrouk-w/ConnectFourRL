from pathlib import Path
import time

from GameEngine.game import ConnectFour
from GameEngine.gui import Gui

from Agents.ppo_agent import PpoAgent


def get_checkpoints():
    project_directory = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    checkpoints_directory = (
        project_directory
        / "Training"
        / "checkpoints"
    )

    checkpoints = []

    for path in checkpoints_directory.iterdir():
        if path.is_dir():
            checkpoints.append(path)

    def get_number(path):
        name = path.name
        number = name.split("_")[-1]

        return int(number)

    checkpoints.sort(key=get_number)

    return checkpoints


def choose_checkpoint(checkpoints):
    print()
    print("Available checkpoints:")
    print()

    for index in range(len(checkpoints)):
        print(
            index + 1,
            "-",
            checkpoints[index].name
        )

    print()

    choice = int(
        input("Choose checkpoint: ")
    )

    return checkpoints[choice - 1]

def play_human_vs_model(checkpoint):
    print()
    print("Loading:", checkpoint.name)

    model = PpoAgent(checkpoint)

    game = ConnectFour()
    gui = Gui()

    print()
    print("Who starts?")
    print("1 - Human")
    print("2 - PPO")

    choice = input("Choice: ")

    if choice == "1":
        human_player = 1
        model_player = 2
    else:
        model_player = 1
        human_player = 2

    game.reset()

    while not game.done:
        gui.render(game.get_state())

        valid_actions = game.get_valid_actions()

        if game.current_player == human_player:
            print("Human - player", human_player)

            action = input("Choose column (0-6): ")

            try:
                action = int(action)

            except ValueError:
                print("Enter a number.")
                continue

            if not game.is_valid_action(action):
                print("Invalid move.")
                continue

        else:
            action = model.select_action(
                game.get_state(),
                valid_actions,
                model_player
            )

            print(
                checkpoint.name,
                "chooses:",
                action
            )

        game.step(action)

    gui.render(game.get_state())

    if game.winner is None:
        print("Draw!")

    elif game.winner == human_player:
        print("Human wins!")

    else:
        print(checkpoint.name, "wins!")

def play_model_vs_model(
    checkpoint_1,
    checkpoint_2
):
    print()
    print(
        checkpoint_1.name,
        "vs",
        checkpoint_2.name
    )

    model_1 = PpoAgent(checkpoint_1)
    model_2 = PpoAgent(checkpoint_2)

    game = ConnectFour()
    gui = Gui()

    game.reset()

    while not game.done:
        gui.render(game.get_state())

        valid_actions = game.get_valid_actions()

        if game.current_player == 1:
            action = model_1.select_action(
                game.get_state(),
                valid_actions,
                1
            )

            print(
                checkpoint_1.name,
                "chooses:",
                action
            )

        else:
            action = model_2.select_action(
                game.get_state(),
                valid_actions,
                2
            )

            print(
                checkpoint_2.name,
                "chooses:",
                action
            )

        game.step(action)

        time.sleep(0.5)

    gui.render(game.get_state())

    if game.winner == 1:
        print(checkpoint_1.name, "wins!")

    elif game.winner == 2:
        print(checkpoint_2.name, "wins!")

    else:
        print("Draw!")

if __name__ == "__main__":
    checkpoints = get_checkpoints()

    print()
    print("Connect Four PPO")
    print()
    print("1 - Human vs checkpoint")
    print("2 - Checkpoint vs checkpoint")

    mode = input("Choose mode: ")

    if mode == "1":
        checkpoint = choose_checkpoint(
            checkpoints
        )

        play_human_vs_model(
            checkpoint
        )

    elif mode == "2":
        print()
        print("Choose Player 1")

        checkpoint_1 = choose_checkpoint(
            checkpoints
        )

        print()
        print("Choose Player 2")

        checkpoint_2 = choose_checkpoint(
            checkpoints
        )

        play_model_vs_model(
            checkpoint_1,
            checkpoint_2
        )

    else:
        print("Invalid mode.")