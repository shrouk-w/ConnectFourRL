import random


class RandomAgent:

    def select_action(self, valid_actions):
        if len(valid_actions) == 0:
            raise ValueError("No valid actions available.")

        return random.choice(valid_actions)