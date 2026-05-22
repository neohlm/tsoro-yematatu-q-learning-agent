"""
play.py — Play Against the Trained Agent
=========================================
Loads the saved Q-table and lets you play Tsoro Yematatu
against the trained agent in the terminal.

Usage:
    python play.py

Run train.py first to generate q_table.pkl.
"""

from game  import TsoroYematatu, PLAYER1, PLAYER2
from agent import QLearningAgent


def human_move(env):
    """Prompt the human player to enter a legal move."""
    legal = env.get_legal_moves()
    env.render()

    while True:
        try:
            if env.phase == "placement":
                print(f"Legal placements: {legal}")
                choice = int(input("Enter position to place your piece: "))
                if choice in legal:
                    return choice
                print("Invalid. Try again.")

            else:
                print("Legal moves (from, to):")
                for i, m in enumerate(legal):
                    print(f"  {i}: {m[0]} → {m[1]}")
                idx = int(input("Enter move number: "))
                if 0 <= idx < len(legal):
                    return legal[idx]
                print("Invalid. Try again.")

        except (ValueError, IndexError):
            print("Please enter a valid number.")


def play():
    env   = TsoroYematatu()
    agent = QLearningAgent()

    try:
        agent.load("q_table.pkl")
        agent.epsilon = 0.0  # no exploration during play
    except FileNotFoundError:
        print("No trained model found. Run train.py first.")
        return

    print("\n" + "="*40)
    print("  Tsoro Yematatu — Play vs Trained Agent")
    print("="*40)
    print("  You are Player 1 (X)")
    print("  Agent is Player 2 (O)")
    print("  Goal: get 3 in a row!\n")

    state = env.reset()

    while not env.done:
        if env.turn == PLAYER1:
            move = human_move(env)
        else:
            legal  = env.get_legal_moves()
            move   = agent.choose_action(state, legal, greedy=True)
            print(f"  Agent plays: {move}")

        state, _, done, info = env.step(move)

    env.render()
    if info["winner"] == PLAYER1:
        print("You win! Congratulations!")
    elif info["winner"] == PLAYER2:
        print("Agent wins! Better luck next time.")
    else:
        print("It's a draw!")


if __name__ == "__main__":
    play()
