"""
agent.py — Q-Learning Agent for Tsoro Yematatu
===============================================
Implements a tabular Q-learning agent that learns to play
Tsoro Yematatu through self-play.

Q-learning update rule:
    Q(s,a) ← Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]

Where:
    α (alpha)   = learning rate    — how fast to update Q values
    γ (gamma)   = discount factor  — how much to value future rewards
    ε (epsilon) = exploration rate — probability of choosing a random move
"""

import random
import pickle
from collections import defaultdict


class QLearningAgent:
    """
    Tabular Q-Learning agent.

    The Q-table maps (state, action) → expected future reward.
    States and actions are stored as hashable Python objects.
    """

    def __init__(
        self,
        alpha=0.1,       # learning rate
        gamma=0.95,      # discount factor
        epsilon=1.0,     # starting exploration rate
        epsilon_min=0.05,# minimum exploration rate
        epsilon_decay=0.9995  # how fast epsilon shrinks each episode
    ):
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table: defaultdict means unseen (state, action) pairs start at 0
        self.q_table = defaultdict(float)

    # ──────────────────────────────────────────────
    # Action selection
    # ──────────────────────────────────────────────

    def choose_action(self, state, legal_moves, greedy=False):
        """
        Choose an action using epsilon-greedy policy.

        greedy=True  → always pick the best known action (used during evaluation)
        greedy=False → explore randomly with probability epsilon (used during training)
        """
        if not legal_moves:
            return None

        # Exploration: pick a random move
        if not greedy and random.random() < self.epsilon:
            return random.choice(legal_moves)

        # Exploitation: pick the move with the highest Q value
        return self._best_action(state, legal_moves)

    def _best_action(self, state, legal_moves):
        """Return the action with the highest Q(state, action) value."""
        best_value  = float("-inf")
        best_action = None

        for action in legal_moves:
            q_val = self.q_table[(state, str(action))]  # str() makes tuples hashable
            if q_val > best_value:
                best_value  = q_val
                best_action = action

        # Break ties randomly so the agent doesn't always pick the same move
        best_actions = [
            a for a in legal_moves
            if self.q_table[(state, str(a))] == best_value
        ]
        return random.choice(best_actions)

    # ──────────────────────────────────────────────
    # Learning (Q-table update)
    # ──────────────────────────────────────────────

    def learn(self, state, action, reward, next_state, next_legal_moves, done):
        """
        Apply the Q-learning update rule after a transition.

        state         : state before the move
        action        : move that was taken
        reward        : reward received after the move
        next_state    : state after the move
        next_legal_moves : legal moves available in next_state
        done          : True if the game ended
        """
        current_q = self.q_table[(state, str(action))]

        if done or not next_legal_moves:
            # Terminal state: no future rewards possible
            target = reward
        else:
            # Bellman equation: reward + discounted best future value
            best_next_q = max(
                self.q_table[(next_state, str(a))] for a in next_legal_moves
            )
            target = reward + self.gamma * best_next_q

        # Update Q value
        self.q_table[(state, str(action))] += self.alpha * (target - current_q)

    def decay_epsilon(self):
        """Reduce epsilon after each episode (less exploration over time)."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ──────────────────────────────────────────────
    # Save / Load
    # ──────────────────────────────────────────────

    def save(self, filepath="q_table.pkl"):
        """Save the Q-table to disk so training can be resumed."""
        with open(filepath, "wb") as f:
            pickle.dump(dict(self.q_table), f)
        print(f"Q-table saved to {filepath} ({len(self.q_table)} entries)")

    def load(self, filepath="q_table.pkl"):
        """Load a previously saved Q-table."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.q_table = defaultdict(float, data)
        print(f"Q-table loaded from {filepath} ({len(self.q_table)} entries)")


class RandomAgent:
    """
    A baseline agent that always picks a random legal move.
    Used to benchmark the Q-learning agent's performance.
    """

    def choose_action(self, state, legal_moves, greedy=False):
        if not legal_moves:
            return None
        return random.choice(legal_moves)

    def learn(self, *args, **kwargs):
        pass  # random agent does not learn

    def decay_epsilon(self):
        pass
