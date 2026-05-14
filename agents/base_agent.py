"""Agent Base.

We define the base class for all agents in this file.
"""
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    def __init__(self):
        """Base agent. All other agents should build on this class.

        As a reminder, you are free to add more methods/functions to this class
        if your agent requires it.
        """

    @abstractmethod
    def take_action(self, state: tuple[int, int]) -> int:
        """Any code that does the action should be included here.

        Args:
            state: The updated position of the agent.
        """
        raise NotImplementedError

    def update(self, state: tuple[int, int], reward: float, action: int):
        """Process a transition. Default is a no-op so non-learning agents
        (e.g. Random, Null) don't need to override it.

        Note: this signature is intentionally narrow (it lacks `next_state`,
        `next_action`, and `done`). TD-style learners (SARSA, Q-learning,
        SARSA(lambda)) need that extra information and therefore expose a
        richer `learn(...)` method called directly by their training loop;
        they override this method to raise so that accidental routing
        through the thin BaseAgent interface fails loudly.

        Args:
            state: The updated position of the agent.
            reward: The value which is returned by the environment as a
                reward.
            action: The action which was taken by the agent.
        """
        return None
