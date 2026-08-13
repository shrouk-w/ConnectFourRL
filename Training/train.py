import ray

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.connectors.env_to_module import FlattenObservations

from Environments.connect_four_gym_env import ConnectFourGymEnv

from pathlib import Path


def create_connector(env, spaces, device):
    return FlattenObservations()


if __name__ == "__main__":

    ray.init()

    config = (
        PPOConfig()

        .environment(
            env=ConnectFourGymEnv
        )

        .env_runners(
            num_env_runners=4,
            num_envs_per_env_runner=8,
            env_to_module_connector=create_connector
        )

        .learners(
            num_learners=0,
            num_gpus_per_learner=1
        )

        .training(
            lr=0.0003,
            gamma=0.99,
            train_batch_size_per_learner=2048,
            minibatch_size=256
        )
    )

    algorithm = config.build_algo()

    for iteration in range(100):
        result = algorithm.train()

        mean_reward = result["env_runners"]["episode_return_mean"]

        print(
            "Iteration:",
            iteration,
            "Mean reward:",
            mean_reward
        )

        if (iteration + 1) % 10 == 0:
            checkpoint_path = (
                    Path("checkpoints")
                    / f"ppo_vs_random_{iteration + 1}"
            )

            checkpoint_uri = checkpoint_path.resolve().as_uri()

            saved_path = algorithm.save_to_path(
                checkpoint_uri
            )

            print("Checkpoint saved:", saved_path)

    algorithm.stop()

    ray.shutdown()