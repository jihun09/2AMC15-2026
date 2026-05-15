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
                 epsilon_decay_mode: str = "multiplicative",
                 epsilon_decay_episodes: int = 1,
                 n_actions: int = 4,
                 rng_seed: int | None = None):
        """Initialize the SARSA agent.

        Args:
            alpha: Learning rate — how much new information overrides old.
            gamma: Discount factor — how much future rewards are valued.
            epsilon: Exploration rate for ε-greedy policy.
            epsilon_decay: Multiplicative decay applied to epsilon after each
                episode (only used when epsilon_decay_mode="multiplicative").
            epsilon_min: Minimum value epsilon can decay to (also serves as
                the final value for linear decay).
            epsilon_decay_mode: "multiplicative" for ε *= decay each episode,
                "linear" for linear interpolation from epsilon to epsilon_min
                over epsilon_decay_episodes, or "fixed" for no decay.
            epsilon_decay_episodes: Number of episodes over which linear decay
                happens (only used when epsilon_decay_mode="linear").
            n_actions: Number of available actions (4: down, up, left, right).
            rng_seed: Seed for the agent's own RNG. If None, uses a random seed.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon
        self._epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.epsilon_decay_mode = epsilon_decay_mode
        self.epsilon_decay_episodes = max(1, epsilon_decay_episodes)
        self.n_actions = n_actions

        # Dedicated RNG for reproducibility without affecting global state
        self._rng = random.Random(rng_seed)

        # Q-table: maps (state, action) -> value
        # Using defaultdict so unseen state-action pairs default to 0.0
        self.q_table = defaultdict(float)

        # Episode counter (used for linear decay)
        self._episode_count = 0

        # These are used by the SARSA training loop (update() interface)
        self.prev_state = None
        self.prev_action = None

    @property
    def epsilon(self) -> float:
        """Current effective exploration rate."""
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value: float):
        self._epsilon = value

    def take_action(self, state: tuple[int, int]) -> int:
        """Select an action using ε-greedy policy.

        With probability ε, pick a random action (explore).
        Otherwise, pick the action with the highest Q-value (exploit).

        Args:
            state: Current position (row, col) of the agent.

        Returns:
            Action integer (0-3).
        """
        if self._rng.random() < self._epsilon:
            return self._rng.randint(0, self.n_actions - 1)
        else:
            # Pick the action with the highest Q-value for this state
            q_values = [self.q_table[(state, a)] for a in range(self.n_actions)]
            max_q = max(q_values)
            # Break ties randomly
            best_actions = [a for a, q in enumerate(q_values) if q == max_q]
            return self._rng.choice(best_actions)

    def select_action(self, state: tuple[int, int], training: bool = True) -> int:
        """Select an action — ε-greedy during training, greedy during eval.

        Args:
            state: Current position (row, col) of the agent.
            training: If True, use ε-greedy. If False, use greedy.

        Returns:
            Action integer (0-3).
        """
        if training:
            return self.take_action(state)
        else:
            q_values = [self.q_table[(state, a)] for a in range(self.n_actions)]
            max_q = max(q_values)
            best_actions = [a for a, q in enumerate(q_values) if q == max_q]
            return self._rng.choice(best_actions)

    def learn(self, state: tuple[int, int], action: int, reward: float,
              next_state: tuple[int, int], next_action: int, done: bool):
        """Apply one SARSA update from a full (S, A, R, S', A') transition.

        This is the preferred training interface — the training loop passes
        all values explicitly.

        Args:
            state: The state the agent was in.
            action: The action that was actually executed.
            reward: The reward received.
            next_state: The state the agent transitioned to.
            next_action: The next action chosen from next_state.
            done: Whether next_state is terminal.
        """
        current_q = self.q_table[(state, action)]

        if done:
            # Terminal state: no future reward to bootstrap from
            target = reward
        else:
            target = reward + self.gamma * self.q_table[(next_state, next_action)]

        td_error = target - current_q
        self.q_table[(state, action)] = current_q + self.alpha * td_error

    def update(self, next_state: tuple[int, int], reward: float, actual_action: int,
               done: bool = False):
        """SARSA update compatible with the base train.py loop.

        This method allows the agent to work with the original train.py
        training loop (which calls agent.update after each step). It tracks
        the previous state/action internally.

        Args:
            next_state: The state the agent ended up in after the step.
            reward: The reward received from the environment.
            actual_action: The action that was actually executed (may differ
                from intended due to stochasticity).
            done: Whether the episode has terminated.
        """
        if self.prev_state is None:
            # First step — just record state and action, no update yet
            self.prev_state = next_state
            self.prev_action = actual_action
            return

        # Choose next action from next_state (on-policy: same ε-greedy)
        next_action = self.take_action(next_state)

        # SARSA update with proper terminal handling
        self.learn(self.prev_state, actual_action, reward,
                   next_state, next_action, done)

        # Move forward
        self.prev_state = next_state
        self.prev_action = next_action

    def decay_epsilon(self):
        """Decay epsilon after an episode. Call this at the end of each episode."""
        self._episode_count += 1

        if self.epsilon_decay_mode == "multiplicative":
            self._epsilon = max(self.epsilon_min,
                                self._epsilon * self.epsilon_decay)
        elif self.epsilon_decay_mode == "linear":
            frac = min(1.0, self._episode_count / self.epsilon_decay_episodes)
            self._epsilon = self.epsilon_start + \
                (self.epsilon_min - self.epsilon_start) * frac
        # "fixed" mode: do nothing

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
