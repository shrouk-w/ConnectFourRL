import gymnasium as gym
import numpy as np

from gymnasium import spaces

from GameEngine.game import ConnectFour
from Agents.random_agent import RandomAgent


class ConnectFourGymEnv(gym.Env):

    def __init__(self, config=None):
        super().__init__()

        self.game = ConnectFour()
        self.opponent = RandomAgent()

        self.action_space = spaces.Discrete(7)

        self.observation_space = spaces.Dict({
            "observation": spaces.Box(
                low=0,
                high=1,
                shape=(6, 7, 2),
                dtype=np.int8
            ),

            "action_mask": spaces.Box(
                low=0,
                high=1,
                shape=(7,),
                dtype=np.int8
            )
        })


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.game.reset()

        observation = self._get_observation()

        info = {}

        return observation, info


    def step(self, action):

        if not self.game.is_valid_action(action):
            observation = self._get_observation()

            reward = -1.0
            terminated = True
            truncated = False

            info = {
                "illegal_action": True
            }

            return (
                observation,
                reward,
                terminated,
                truncated,
                info
            )

        self.game.step(action)

        if self.game.done:
            reward = self._get_reward()

            return (
                self._get_observation(),
                reward,
                True,
                False,
                {}
            )

        valid_actions = self.game.get_valid_actions()

        opponent_action = self.opponent.select_action(
            valid_actions
        )

        self.game.step(opponent_action)

        if self.game.done:
            reward = self._get_reward()

            return (
                self._get_observation(),
                reward,
                True,
                False,
                {}
            )

        return (
            self._get_observation(),
            0.0,
            False,
            False,
            {}
        )


    def _get_observation(self):
        board = self.game.get_state()

        observation = np.zeros(
            (6, 7, 2),
            dtype=np.int8
        )

        for row in range(6):
            for column in range(7):

                if board[row][column] == 1:
                    observation[row][column][0] = 1

                elif board[row][column] == 2:
                    observation[row][column][1] = 1

        action_mask = self.game.get_action_mask()

        return {
            "observation": observation,
            "action_mask": action_mask
        }


    def _get_reward(self):

        if self.game.winner == 1:
            return 1.0

        if self.game.winner == 2:
            return -1.0

        return 0.0