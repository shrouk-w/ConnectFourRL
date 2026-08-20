from pathlib import Path

from GameEngine.game import ConnectFour

from Agents.ppo_agent import PpoAgent
from Agents.random_agent import RandomAgent
from Agents.heuristic_agent import HeuristicAgent


GAMES_VS_BASELINE = 200
GAMES_VS_CHECKPOINT = 1000


def get_checkpoints():
    training_directory = Path(__file__).resolve().parent

    checkpoints_directory = (
        training_directory
        / "checkpoints"
    )

    checkpoints = []

    for path in checkpoints_directory.iterdir():
        if path.is_dir():
            checkpoints.append(path)

    def get_number(path):
        number = path.name.split("_")[-1]

        return int(number)

    checkpoints.sort(key=get_number)

    return checkpoints


def get_action(
    agent,
    agent_type,
    game,
    player
):
    state = game.get_state()
    valid_actions = game.get_valid_actions()

    if agent_type == "ppo":
        return agent.select_action(
            state,
            valid_actions,
            player,
            deterministic=False
        )

    if agent_type == "random":
        return agent.select_action(
            valid_actions
        )

    if agent_type == "heuristic":
        return agent.select_action(
            state,
            valid_actions,
            player
        )

    raise ValueError(
        "Unknown agent type: " + agent_type
    )


def play_game(
    player_1_agent,
    player_1_type,
    player_2_agent,
    player_2_type
):
    game = ConnectFour()

    game.reset()

    while not game.done:

        if game.current_player == 1:

            action = get_action(
                player_1_agent,
                player_1_type,
                game,
                1
            )

        else:

            action = get_action(
                player_2_agent,
                player_2_type,
                game,
                2
            )

        game.step(action)

    return game.winner

def evaluate_agents(
    agent_a,
    type_a,
    agent_b,
    type_b,
    games
):
    wins_a = 0
    wins_b = 0
    draws = 0

    for game_number in range(games):

        # Co drugą grę zamieniamy strony.
        if game_number % 2 == 0:

            winner = play_game(
                agent_a,
                type_a,
                agent_b,
                type_b
            )

            if winner == 1:
                wins_a += 1

            elif winner == 2:
                wins_b += 1

            else:
                draws += 1

        else:

            winner = play_game(
                agent_b,
                type_b,
                agent_a,
                type_a
            )

            if winner == 1:
                wins_b += 1

            elif winner == 2:
                wins_a += 1

            else:
                draws += 1

    return wins_a, draws, wins_b

def calculate_score(
    wins,
    draws,
    losses
):
    total_games = (
        wins
        + draws
        + losses
    )

    total_reward = (
        wins * 1.0
        + draws * -0.1
        + losses * -1.0
    )

    return total_reward / total_games


def evaluate_against_baselines(
    checkpoints
):
    random_agent = RandomAgent()
    heuristic_agent = HeuristicAgent()

    results = []

    print()
    print("================================")
    print("BASELINE EVALUATION")
    print("================================")

    for checkpoint in checkpoints:

        print()
        print("Loading:", checkpoint.name)

        model = PpoAgent(checkpoint)

        random_result = evaluate_agents(
            model,
            "ppo",
            random_agent,
            "random",
            GAMES_VS_BASELINE
        )

        heuristic_result = evaluate_agents(
            model,
            "ppo",
            heuristic_agent,
            "heuristic",
            GAMES_VS_BASELINE
        )

        random_wins = random_result[0]
        random_draws = random_result[1]
        random_losses = random_result[2]

        heuristic_wins = heuristic_result[0]
        heuristic_draws = heuristic_result[1]
        heuristic_losses = heuristic_result[2]

        random_score = calculate_score(
            random_wins,
            random_draws,
            random_losses
        )

        heuristic_score = calculate_score(
            heuristic_wins,
            heuristic_draws,
            heuristic_losses
        )

        combined_score = (
            random_score
            + heuristic_score
        ) / 2

        results.append({
            "name": checkpoint.name,
            "random_score": random_score,
            "heuristic_score": heuristic_score,
            "score": combined_score
        })

        print(
            "vs Random:",
            random_wins,
            "/",
            random_draws,
            "/",
            random_losses,
            "score:",
            round(random_score, 3)
        )

        print(
            "vs Heuristic:",
            heuristic_wins,
            "/",
            heuristic_draws,
            "/",
            heuristic_losses,
            "score:",
            round(heuristic_score, 3)
        )

    results.sort(
        key=lambda result: result["score"],
        reverse=True
    )

    print()
    print("================================")
    print("BASELINE RANKING")
    print("================================")

    for index in range(len(results)):

        result = results[index]

        print(
            index + 1,
            result["name"],
            "score:",
            round(result["score"], 3),
            "| random:",
            round(result["random_score"], 3),
            "| heuristic:",
            round(result["heuristic_score"], 3)
        )

    return results

def checkpoint_tournament(
    checkpoints
):
    print()
    print("================================")
    print("CHECKPOINT TOURNAMENT")
    print("================================")

    models = {}

    scores = {}

    for checkpoint in checkpoints:

        print(
            "Loading:",
            checkpoint.name
        )

        models[checkpoint.name] = PpoAgent(
            checkpoint
        )

        scores[checkpoint.name] = 0.0


    for i in range(len(checkpoints)):

        for j in range(
            i + 1,
            len(checkpoints)
        ):

            checkpoint_a = checkpoints[i]
            checkpoint_b = checkpoints[j]

            name_a = checkpoint_a.name
            name_b = checkpoint_b.name

            model_a = models[name_a]
            model_b = models[name_b]

            print()
            print(
                name_a,
                "vs",
                name_b
            )

            wins_a, draws, wins_b = (
                evaluate_agents(
                    model_a,
                    "ppo",
                    model_b,
                    "ppo",
                    GAMES_VS_CHECKPOINT
                )
            )

            score_a = calculate_score(
                wins_a,
                draws,
                wins_b
            )

            score_b = calculate_score(
                wins_b,
                draws,
                wins_a
            )

            scores[name_a] += score_a
            scores[name_b] += score_b

            print(
                wins_a,
                "-",
                draws,
                "-",
                wins_b
            )


    ranking = []

    for name in scores:

        ranking.append({
            "name": name,
            "score": scores[name]
        })


    def get_score(result):
        return result["score"]


    ranking.sort(
        key=get_score,
        reverse=True
    )


    print()
    print("================================")
    print("TOURNAMENT RANKING")
    print("================================")

    for index in range(len(ranking)):

        result = ranking[index]

        print(
            index + 1,
            result["name"],
            "score:",
            round(result["score"], 3)
        )

if __name__ == "__main__":

    checkpoints = get_checkpoints()

    print()
    print("Connect Four Evaluator")
    print()
    print(
        "Found",
        len(checkpoints),
        "checkpoints"
    )

    print()
    print("1 - Evaluate vs Random + Heuristic")
    print("2 - Checkpoint tournament")
    print("3 - Both")

    mode = input("Choose mode: ")

    if mode == "1":

        evaluate_against_baselines(
            checkpoints
        )

    elif mode == "2":

        checkpoint_tournament(
            checkpoints
        )

    elif mode == "3":

        evaluate_against_baselines(
            checkpoints
        )

        checkpoint_tournament(
            checkpoints
        )

    else:

        print("Invalid mode.")