"""SARSA Agent.

An on-policy TD(0) agent that learns Q-values using the SARSA update rule.
"""
import random
import numpy as np
from collections import defaultdict

from agents import BaseAgent


class SarsaAgent(BaseAgent):
    """SARSA (State-Action-Reward-State-Action) agent.

    Uses an ε-greedy policy for exploration and updates Q-values
    based on the actual next action chosen (on-policy).
    """

    def __init__(self,
                 alpha: float = 0.1,
                 gamma: float = 0.9,
                 epsilon: float = 0.1,
                 epsilon_decay: float = 1.0,
                 epsilon_min: float = 0.01,
                 n_actions: int = 4):
        """Initialize the SARSA agent.

        Args:
            alpha: Learning rate — how much new information overrides old.
            gamma: Discount factor — how much future rewards are valued.
            epsilon: Exploration rate for ε-greedy policy.
            epsilon_decay: Multiplicative decay applied to epsilon after each episode.
            epsilon_min: Minimum value epsilon can decay to.
            n_actions: Number of available actions (4: down, up, left, right).
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.n_actions = n_actions

        # Q-table: maps (state, action) -> value
        # Using defaultdict so unseen state-action pairs default to 0.0
        self.q_table = defaultdict(float)

        # These are used by the SARSA training loop
        self.prev_state = None
        self.prev_action = None

    def take_action(self, state: tuple[int, int]) -> int:
        """Select an action using ε-greedy policy.

        With probability ε, pick a random action (explore).
        Otherwise, pick the action with the highest Q-value (exploit).

        Args:
            state: Current position (row, col) of the agent.

        Returns:
            Action integer (0-3).
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            # Pick the action with the highest Q-value for this state
            q_values = [self.q_table[(state, a)] for a in range(self.n_actions)]
            max_q = max(q_values)
            # Break ties randomly
            best_actions = [a for a, q in enumerate(q_values) if q == max_q]
            return random.choice(best_actions)

    def update(self, next_state: tuple[int, int], reward: float, actual_action: int):
        """SARSA update compatible with the base train.py loop.

        This method allows the agent to work with the original train.py
        training loop (which calls agent.update after each step). It tracks
        the previous state/action internally.

        For the dedicated train_sarsa.py, the update is done inline for clarity.

        Args:
            next_state: The state the agent ended up in after the step.
            reward: The reward received from the environment.
            actual_action: The action that was actually executed (may differ
                from intended due to stochasticity).
        """
        if self.prev_state is None:
            # First step — just record state and action, no update yet
            self.prev_state = next_state
            self.prev_action = actual_action
            return

        # Choose next action from next_state (on-policy: same ε-greedy)
        next_action = self.take_action(next_state)

        # SARSA update: Q(s,a) += α * [r + γ*Q(s',a') - Q(s,a)]
        current_q = self.q_table[(self.prev_state, actual_action)]
        next_q = self.q_table[(next_state, next_action)]
        td_error = reward + self.gamma * next_q - current_q
        self.q_table[(self.prev_state, actual_action)] = current_q + self.alpha * td_error

        # Move forward
        self.prev_state = next_state
        self.prev_action = next_action

    def decay_epsilon(self):
        """Decay epsilon after an episode. Call this at the end of each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def reset_episode(self):
        """Reset episode-specific state. Call at the start of each new episode."""
        self.prev_state = None
        self.prev_action = None

    def get_policy(self) -> dict:
        """Extract the greedy policy from the Q-table.

        Returns:
            Dictionary mapping state -> best action.
        """
        policy = {}
        states = set(s for (s, a) in self.q_table.keys())
        for state in states:
            q_values = [self.q_table[(state, a)] for a in range(self.n_actions)]
            policy[state] = int(np.argmax(q_values))
        return policy
