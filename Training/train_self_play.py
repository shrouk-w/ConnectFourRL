from pathlib import Path

import ray

from ray.tune.registry import register_env

from ray.rllib.algorithms.ppo import PPOConfig

from ray.rllib.env.wrappers.pettingzoo_env import (
    PettingZooEnv
)

from ray.rllib.core.rl_module.rl_module import (
    RLModuleSpec
)

from ray.rllib.core.rl_module.multi_rl_module import (
    MultiRLModuleSpec
)

from ray.rllib.core.rl_module.default_model_config import (
    DefaultModelConfig
)

from ray.rllib.examples.rl_modules.classes.action_masking_rlm import (
    ActionMaskingTorchRLModule
)

from Environments.connect_four_env import create_env

from Agents.rllib_opponents import (
    RandomAgentRLModule,
    HeuristicAgentRLModule
)


def create_rllib_env(config):
    environment = create_env()

    return PettingZooEnv(environment)


register_env(
    "connect_four_self_play",
    create_rllib_env
)

def choose_opponent(episode):
    episode_hash = abs(hash(episode.id_))

    roll = episode_hash % 100

    if roll < 25:
        return "self"

    if roll < 50:
        return "previous"

    if roll < 75:
        return "random"

    return "heuristic"

def get_main_player(episode):
    episode_hash = abs(hash(episode.id_))

    value = episode_hash // 100

    if value % 2 == 0:
        return "player_0"

    return "player_1"

def policy_mapping(agent_id, episode, **kwargs):
    opponent = choose_opponent(episode)

    if opponent == "self":
        return "main"

    main_player = get_main_player(episode)

    if agent_id == main_player:
        return "main"

    if opponent == "previous":
        return "previous"

    if opponent == "random":
        return "random"

    if opponent == "heuristic":
        return "heuristic"

    raise ValueError("Unknown opponent.")

if __name__ == "__main__":

    ray.init()

    config = (
        PPOConfig()

        .framework("torch")

        .environment(
            env="connect_four_self_play"
        )

        .env_runners(
            num_env_runners=4,
            num_envs_per_env_runner=1
        )

        .learners(
            num_learners=0,
            num_gpus_per_learner=1
        )

        .multi_agent(
            policies={
                "main",
                "previous",
                "random",
                "heuristic"
            },

            policy_mapping_fn=policy_mapping,

            policies_to_train=[
                "main"
            ]
        )

        .rl_module(
            model_config=DefaultModelConfig(
                fcnet_hiddens=[
                    256,
                    256
                ],
                fcnet_activation="relu"
            ),

            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={

                    "main": RLModuleSpec(
                        module_class=ActionMaskingTorchRLModule
                    ),

                    "previous": RLModuleSpec(
                        module_class=ActionMaskingTorchRLModule
                    ),

                    "random": RLModuleSpec(
                        module_class=RandomAgentRLModule
                    ),

                    "heuristic": RLModuleSpec(
                        module_class=HeuristicAgentRLModule
                    )
                }
            )
        )

        .training(
            lr=0.0003,
            gamma=0.99,
            train_batch_size_per_learner=4096,
            minibatch_size=256
        )
    )

    algorithm = config.build_algo()

    main_weights = algorithm.get_weights(
        ["main"]
    )

    algorithm.set_weights({
        "previous": main_weights["main"]
    })

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

    checkpoints_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    for iteration in range(1, 1001):
        result = algorithm.train()

        print(
            "Iteration:",
            iteration
        )

        if iteration % 100 == 0:

            main_weights = algorithm.get_weights(
                ["main"]
            )

            algorithm.set_weights({
                "previous": main_weights["main"]
            })

            print(
                "Updated previous model at iteration",
                iteration
            )

            checkpoint_directory = (
                    checkpoints_directory
                    / f"self_play_{iteration}"
            )

            checkpoint_uri = (
                checkpoint_directory
                .resolve()
                .as_uri()
            )

            saved_path = algorithm.save_to_path(
                checkpoint_uri
            )

            print(
                "Saved checkpoint:",
                saved_path
            )

    algorithm.stop()

    ray.shutdown()