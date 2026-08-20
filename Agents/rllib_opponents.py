import numpy as np

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module import RLModule
from ray.rllib.utils.annotations import override

from Agents.random_agent import RandomAgent
from Agents.heuristic_agent import HeuristicAgent


class RandomAgentRLModule(RLModule):

    def setup(self):
        self.agent = RandomAgent()

    @override(RLModule)
    def _forward(self, batch, **kwargs):
        masks = batch[Columns.OBS]["action_mask"]

        actions = []

        for i in range(len(masks)):
            mask = self._to_numpy(masks[i])

            valid_actions = []

            for action in range(7):
                if mask[action] == 1:
                    valid_actions.append(action)

            selected_action = self.agent.select_action(
                valid_actions
            )

            actions.append(selected_action)

        return {
            Columns.ACTIONS: np.array(
                actions,
                dtype=np.int64
            )
        }

    @override(RLModule)
    def _forward_train(self, *args, **kwargs):
        raise NotImplementedError(
            "RandomAgentRLModule should not be trained."
        )

    def _to_numpy(self, value):
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()

        return np.asarray(value)


class HeuristicAgentRLModule(RLModule):

    def setup(self):
        self.agent = HeuristicAgent()

    @override(RLModule)
    def _forward(self, batch, **kwargs):
        observations = batch[Columns.OBS]["observations"]
        masks = batch[Columns.OBS]["action_mask"]

        actions = []

        for i in range(len(observations)):
            observation = self._to_numpy(
                observations[i]
            )

            mask = self._to_numpy(
                masks[i]
            )

            board = self._observation_to_board(
                observation
            )

            valid_actions = []

            for action in range(7):
                if mask[action] == 1:
                    valid_actions.append(action)

            selected_action = self.agent.select_action(
                board,
                valid_actions,
                1
            )

            actions.append(selected_action)

        return {
            Columns.ACTIONS: np.array(
                actions,
                dtype=np.int64
            )
        }

    @override(RLModule)
    def _forward_train(self, *args, **kwargs):
        raise NotImplementedError(
            "HeuristicAgentRLModule should not be trained."
        )

    def _observation_to_board(self, observation):
        board = np.zeros(
            (6, 7),
            dtype=np.int8
        )

        for row in range(6):
            for column in range(7):
                index = row * 7 + column

                if observation[index] == 1:
                    board[row][column] = 1

                elif observation[42 + index] == 1:
                    board[row][column] = 2

        return board

    def _to_numpy(self, value):
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()

        return np.asarray(value)