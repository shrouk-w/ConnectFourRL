from pathlib import Path

import numpy as np
import torch

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule


class PpoAgent:

    def __init__(self, checkpoint_path):
        checkpoint_path = Path(checkpoint_path)

        module_path = (
            checkpoint_path
            / "learner_group"
            / "learner"
            / "rl_module"
            / "main"
        )

        if not module_path.exists():
            raise FileNotFoundError(
                "Nie znaleziono modelu main w: " + str(module_path)
            )

        module_uri = module_path.resolve().as_uri()

        self.module = RLModule.from_checkpoint(module_uri)

        try:
            self.device = next(self.module.parameters()).device
        except StopIteration:
            self.device = torch.device("cpu")

    def select_action(
            self,
            state,
            valid_actions,
            player,
            deterministic=True
    ):
        observation = self._create_observation(
            state,
            valid_actions,
            player
        )

        observations_tensor = torch.tensor(
            observation["observations"],
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        action_mask_tensor = torch.tensor(
            observation["action_mask"],
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        batch = {
            Columns.OBS: {
                "observations": observations_tensor,
                "action_mask": action_mask_tensor
            }
        }

        with torch.no_grad():
            output = self.module.forward_inference(batch)

        logits = output[Columns.ACTION_DIST_INPUTS][0]

        if deterministic:
            action = torch.argmax(logits).item()

        else:
            distribution = torch.distributions.Categorical(
                logits=logits
            )

            action = distribution.sample().item()

        return action


    def _create_observation(self, state, valid_actions, player):
        observations = np.zeros(
            84,
            dtype=np.float32
        )

        if player == 1:
            opponent = 2
        else:
            opponent = 1

        for row in range(6):
            for column in range(7):
                index = row * 7 + column

                if state[row][column] == player:
                    observations[index] = 1

                elif state[row][column] == opponent:
                    observations[42 + index] = 1

        action_mask = np.zeros(
            7,
            dtype=np.float32
        )

        for action in valid_actions:
            action_mask[action] = 1

        return {
            "observations": observations,
            "action_mask": action_mask
        }