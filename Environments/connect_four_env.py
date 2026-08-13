import numpy as np

from gymnasium import spaces

from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import AgentSelector
from pettingzoo.utils import wrappers

from GameEngine.game import ConnectFour


class ConnectFourEnv(AECEnv):

    metadata = {
        "name": "connect_four_custom_v0"
    }

    def __init__(self):
        super().__init__()

        self.game = ConnectFour()

        self.possible_agents = [
            "player_0",
            "player_1"
        ]

        self.agents = []

        self._action_space = spaces.Discrete(7)

        self._observation_space = spaces.Dict({
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


    def action_space(self, agent):
        return self._action_space


    def observation_space(self, agent):
        return self._observation_space


    def reset(self, seed=None, options=None):
        self.game.reset()

        self.agents = self.possible_agents.copy()

        self.rewards = {
            "player_0": 0.0,
            "player_1": 0.0
        }

        self._cumulative_rewards = {
            "player_0": 0.0,
            "player_1": 0.0
        }

        self.terminations = {
            "player_0": False,
            "player_1": False
        }

        self.truncations = {
            "player_0": False,
            "player_1": False
        }

        self.infos = {
            "player_0": {},
            "player_1": {}
        }

        self._agent_selector = AgentSelector(self.agents)

        self.agent_selection = self._agent_selector.reset()


    def observe(self, agent):
        board = self.game.get_state()

        observation = np.zeros(
            (6, 7, 2),
            dtype=np.int8
        )

        if agent == "player_0":
            player_number = 1
            opponent_number = 2
        else:
            player_number = 2
            opponent_number = 1

        for row in range(6):
            for column in range(7):

                if board[row][column] == player_number:
                    observation[row][column][0] = 1

                elif board[row][column] == opponent_number:
                    observation[row][column][1] = 1

        action_mask = np.zeros(7, dtype=np.int8)

        if agent == self.agent_selection and not self.game.done:
            action_mask = self.game.get_action_mask()

        return {
            "observation": observation,
            "action_mask": action_mask
        }


    def step(self, action):
        current_agent = self.agent_selection

        if (
            self.terminations[current_agent]
            or self.truncations[current_agent]
        ):
            self._was_dead_step(action)
            return

        self._cumulative_rewards[current_agent] = 0

        self._clear_rewards()

        self.game.step(action)

        next_agent = self._agent_selector.next()

        if self.game.done:
            self._set_final_rewards()

            self.terminations["player_0"] = True
            self.terminations["player_1"] = True

        self.agent_selection = next_agent

        self._accumulate_rewards()


    def _set_final_rewards(self):
        if self.game.winner == 1:
            self.rewards["player_0"] = 1.0
            self.rewards["player_1"] = -1.0

        elif self.game.winner == 2:
            self.rewards["player_0"] = -1.0
            self.rewards["player_1"] = 1.0

        else:
            self.rewards["player_0"] = -0.1
            self.rewards["player_1"] = -0.1


def create_env():
    environment = ConnectFourEnv()

    environment = wrappers.TerminateIllegalWrapper(
        environment,
        illegal_reward=-1
    )

    environment = wrappers.AssertOutOfBoundsWrapper(
        environment
    )

    environment = wrappers.OrderEnforcingWrapper(
        environment
    )

    return environment