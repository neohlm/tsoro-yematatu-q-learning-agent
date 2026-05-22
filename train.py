"""
train.py — Training Script for the Tsoro Yematatu Q-Learning Agent
===================================================================
Runs self-play training and periodically evaluates the agent's
win rate against a random opponent. Saves results for plotting.

Usage:
    python train.py

Output:
    q_table.pkl         — saved Q-table (reloadable)
    training_log.pkl    — win/draw/loss rates per evaluation checkpoint
"""

import pickle
import random
from game  import TsoroYematatu, PLAYER1, PLAYER2
from agent import QLearningAgent, RandomAgent


# ──────────────────────────────────────────────
# Hyperparameters — tweak these if you like
# ──────────────────────────────────────────────

TOTAL_EPISODES   = 20_000   # total self-play training games
EVAL_INTERVAL    = 500      # evaluate every N episodes
EVAL_GAMES       = 200      # games per evaluation checkpoint
MAX_MOVES        = 200      # max moves per game (prevents infinite loops)

ALPHA            = 0.1      # learning rate
GAMMA            = 0.95     # discount factor
EPSILON_START    = 1.0      # start fully exploring
EPSILON_MIN      = 0.05     # never go below 5% exploration
EPSILON_DECAY    = 0.9995   # decay rate per episode


# ──────────────────────────────────────────────
# Self-play training loop
# ──────────────────────────────────────────────

def run_episode(env, agent1, agent2, training=True):
    """
    Run one complete game between agent1 (Player 1) and agent2 (Player 2).

    During training: both agents learn from the experience.
    During eval: agents act greedily (no exploration, no learning).

    Returns: winner (1, 2, or None for draw)
    """
    state = env.reset()
    agents = {PLAYER1: agent1, PLAYER2: agent2}

    prev = {
        PLAYER1: {"state": None, "action": None},
        PLAYER2: {"state": None, "action": None},
    }

    for _ in range(MAX_MOVES):
        current = env.turn
        agent   = agents[current]

        legal_moves = env.get_legal_moves()
        if not legal_moves:
            break

        action = agent.choose_action(state, legal_moves, greedy=not training)

        # Store the state/action before stepping (needed for Q update)
        prev[current]["state"]  = state
        prev[current]["action"] = action

        next_state, reward, done, info = env.step(action)

        # ── Q-learning update for the player who just moved ──
        if training:
            next_legal = env.get_legal_moves() if not done else []
            agent.learn(state, action, reward, next_state, next_legal, done)

            # ── Penalise the opponent if this player just won ──
            opponent = PLAYER2 if current == PLAYER1 else PLAYER1
            if done and info["winner"] == current:
                opp_agent = agents[opponent]
                if prev[opponent]["state"] is not None:
                    opp_agent.learn(
                        prev[opponent]["state"],
                        prev[opponent]["action"],
                        reward=-1,          # opponent gets -1 for losing
                        next_state=next_state,
                        next_legal_moves=[],
                        done=True,
                    )

        state = next_state

        if done:
            return info["winner"]

    return None  # draw / timeout


def evaluate(env, agent, n_games=200):
    """
    Evaluate the trained agent against a random opponent.
    Agent always plays as Player 1.

    Returns: (win_rate, draw_rate, loss_rate)
    """
    random_agent = RandomAgent()
    wins = draws = losses = 0

    for _ in range(n_games):
        winner = run_episode(env, agent, random_agent, training=False)
        if winner == PLAYER1:
            wins   += 1
        elif winner is None:
            draws  += 1
        else:
            losses += 1

    total = n_games
    return wins / total, draws / total, losses / total


# ──────────────────────────────────────────────
# Main training routine
# ──────────────────────────────────────────────

def train():
    env    = TsoroYematatu()
    agent  = QLearningAgent(
        alpha         = ALPHA,
        gamma         = GAMMA,
        epsilon       = EPSILON_START,
        epsilon_min   = EPSILON_MIN,
        epsilon_decay = EPSILON_DECAY,
    )

    # The opponent during self-play is also a Q-learning agent
    # (both agents share the same Q-table — common for symmetric games)
    opponent = agent

    training_log = {
        "episodes":  [],
        "win_rate":  [],
        "draw_rate": [],
        "loss_rate": [],
        "epsilon":   [],
    }

    print("=" * 55)
    print("  Tsoro Yematatu — Q-Learning Training")
    print("=" * 55)
    print(f"  Episodes   : {TOTAL_EPISODES:,}")
    print(f"  Eval every : {EVAL_INTERVAL:,} episodes ({EVAL_GAMES} games)")
    print(f"  Alpha      : {ALPHA}  |  Gamma: {GAMMA}")
    print("=" * 55)

    for episode in range(1, TOTAL_EPISODES + 1):

        # Randomly swap who plays first to ensure balanced learning
        if random.random() < 0.5:
            run_episode(env, agent, opponent, training=True)
        else:
            run_episode(env, opponent, agent, training=True)

        agent.decay_epsilon()

        # ── Periodic evaluation ──
        if episode % EVAL_INTERVAL == 0:
            win_rate, draw_rate, loss_rate = evaluate(env, agent, EVAL_GAMES)

            training_log["episodes"].append(episode)
            training_log["win_rate"].append(win_rate)
            training_log["draw_rate"].append(draw_rate)
            training_log["loss_rate"].append(loss_rate)
            training_log["epsilon"].append(agent.epsilon)

            print(
                f"  Episode {episode:>6,} | "
                f"Win: {win_rate*100:5.1f}% | "
                f"Draw: {draw_rate*100:4.1f}% | "
                f"Loss: {loss_rate*100:5.1f}% | "
                f"ε={agent.epsilon:.3f} | "
                f"Q-entries: {len(agent.q_table):,}"
            )

    print("\nTraining complete.")

    # ── Save results ──
    agent.save("q_table.pkl")
    with open("training_log.pkl", "wb") as f:
        pickle.dump(training_log, f)
    print("Training log saved to training_log.pkl")

    return agent, training_log


if __name__ == "__main__":
    trained_agent, log = train()
