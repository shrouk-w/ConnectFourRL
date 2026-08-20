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
from ray.rllib.core.rl_module.rl_module import RLModule

ANCHOR_IDS = [
    "anchor_500",
    "anchor_800",
    "anchor_900",
    "anchor_1000"
]

RECENT_IDS = [
    "recent_0",
    "recent_1",
    "recent_2",
    "recent_3",
    "recent_4",
    "recent_5"
]

POOL_IDS = ANCHOR_IDS + RECENT_IDS

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

    if roll < 20:
        return "self"

    if roll < 25:
        return "random"

    if roll < 40:
        return "heuristic"

    pool_index = (episode_hash // 100) % len(POOL_IDS)

    return POOL_IDS[pool_index]

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

    return opponent


ppo_modules = {
    "main": RLModuleSpec(
        module_class=ActionMaskingTorchRLModule
    )
}

for module_id in POOL_IDS:
    ppo_modules[module_id] = RLModuleSpec(
        module_class=ActionMaskingTorchRLModule
    )

ppo_modules["random"] = RLModuleSpec(
    module_class=RandomAgentRLModule
)

ppo_modules["heuristic"] = RLModuleSpec(
    module_class=HeuristicAgentRLModule
)

def load_main_state(checkpoint_path):
    module_path = (
        checkpoint_path
        / "learner_group"
        / "learner"
        / "rl_module"
        / "main"
    )

    module_uri = module_path.resolve().as_uri()

    module = RLModule.from_checkpoint(
        module_uri
    )

    return module.get_state()

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
            policies=set(ppo_modules.keys()),

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
                rl_module_specs=ppo_modules
            )
        )

        .training(
            lr=0.0001,
            gamma=0.99,

            entropy_coeff=0.005,
            clip_param=0.2,

            train_batch_size_per_learner=4096,
            minibatch_size=256
        )
    )

    algorithm = config.build_algo()

    project_directory = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    old_checkpoints = (
            project_directory
            / "Training"
            / "checkpoints"
    )

    state_500 = load_main_state(
        old_checkpoints / "self_play_500"
    )

    state_800 = load_main_state(
        old_checkpoints / "self_play_800"
    )

    state_900 = load_main_state(
        old_checkpoints / "self_play_900"
    )

    state_1000 = load_main_state(
        old_checkpoints / "self_play_1000"
    )

    initial_weights = {
        "main": state_1000,

        "anchor_500": state_500,
        "anchor_800": state_800,
        "anchor_900": state_900,
        "anchor_1000": state_1000
    }

    for module_id in RECENT_IDS:
        initial_weights[module_id] = state_1000

    algorithm.set_weights(
        initial_weights
    )

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

    for iteration in range(1001, 2001):
        result = algorithm.train()

        print(
            "Iteration:",
            iteration
        )

        if iteration % 100 == 0:
            main_weights = algorithm.get_weights(
                ["main"]
            )

            snapshot_number = (
                    iteration // 100
            )

            recent_index = (
                    snapshot_number
                    % len(RECENT_IDS)
            )

            recent_id = RECENT_IDS[
                recent_index
            ]

            algorithm.set_weights({
                recent_id: main_weights["main"]
            })

            checkpoint_directory = (
                    checkpoints_directory
                    / f"pool_{iteration}"
            )

            checkpoint_uri = (
                checkpoint_directory
                .resolve()
                .as_uri()
            )

            algorithm.save_to_path(
                checkpoint_uri
            )

            print(
                "Updated opponent:",
                recent_id,
                "with main from iteration",
                iteration
            )

    algorithm.stop()

    ray.shutdown()