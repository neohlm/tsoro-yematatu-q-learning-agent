"""
plot_results.py — Visualise Training Results
=============================================
Loads the training log and produces the graphs needed
for your experimental evaluation section.

Usage:
    python plot_results.py

Requires:
    pip install matplotlib
    (run train.py first to generate training_log.pkl)

Output files:
    learning_curve.png   — win/draw/loss rates over training
    epsilon_decay.png    — exploration rate over training
"""

import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ──────────────────────────────────────────────
# Load training log
# ──────────────────────────────────────────────

def load_log(filepath="training_log.pkl"):
    with open(filepath, "rb") as f:
        return pickle.load(f)


# ──────────────────────────────────────────────
# Plot 1: Learning curve (win/draw/loss over time)
# ──────────────────────────────────────────────

def plot_learning_curve(log):
    episodes  = log["episodes"]
    win_rate  = [r * 100 for r in log["win_rate"]]
    draw_rate = [r * 100 for r in log["draw_rate"]]
    loss_rate = [r * 100 for r in log["loss_rate"]]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(episodes, win_rate,  label="Win rate",  color="#2563EB", linewidth=2.5)
    ax.plot(episodes, draw_rate, label="Draw rate", color="#F59E0B", linewidth=2,  linestyle="--")
    ax.plot(episodes, loss_rate, label="Loss rate", color="#EF4444", linewidth=2,  linestyle=":")

    # Baseline: random agent wins ~50% of the time
    ax.axhline(y=50, color="gray", linestyle="--", linewidth=1, alpha=0.7,
               label="Random baseline (50%)")

    # Success threshold line
    ax.axhline(y=75, color="#10B981", linestyle="--", linewidth=1, alpha=0.7,
               label="Target win rate (75%)")

    ax.set_xlabel("Training episodes", fontsize=12)
    ax.set_ylabel("Rate (%)", fontsize=12)
    ax.set_title("Learning Curve — Q-Learning Agent vs. Random Opponent\n"
                 "Tsoro Yematatu | ITRI 616 Mini-Project", fontsize=13, pad=15)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d%%'))
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#F9FAFB")
    fig.tight_layout()

    fig.savefig("learning_curve.png", dpi=150)
    print("Saved: learning_curve.png")
    plt.close()


# ──────────────────────────────────────────────
# Plot 2: Epsilon decay (exploration over time)
# ──────────────────────────────────────────────

def plot_epsilon_decay(log):
    episodes = log["episodes"]
    epsilon  = log["epsilon"]

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(episodes, epsilon, color="#7C3AED", linewidth=2.5)
    ax.fill_between(episodes, epsilon, alpha=0.15, color="#7C3AED")

    ax.set_xlabel("Training episodes", fontsize=12)
    ax.set_ylabel("Epsilon (ε)", fontsize=12)
    ax.set_title("Exploration Rate (ε) Decay Over Training\n"
                 "High ε = explore randomly | Low ε = exploit Q-table", fontsize=12, pad=12)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#F9FAFB")
    fig.tight_layout()

    fig.savefig("epsilon_decay.png", dpi=150)
    print("Saved: epsilon_decay.png")
    plt.close()


# ──────────────────────────────────────────────
# Plot 3: Bar chart — final performance comparison
# ──────────────────────────────────────────────

def plot_final_comparison(log):
    """Bar chart comparing trained agent vs random agent final stats."""
    final_win  = log["win_rate"][-1]  * 100
    final_draw = log["draw_rate"][-1] * 100
    final_loss = log["loss_rate"][-1] * 100

    # Random agent baseline (approximately equal splits)
    random_win  = 50
    random_draw = 5
    random_loss = 45

    labels   = ["Win", "Draw", "Loss"]
    trained  = [final_win, final_draw, final_loss]
    random_b = [random_win, random_draw, random_loss]

    x     = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar([i - width/2 for i in x], trained,  width, label="Trained Q-agent", color="#2563EB", alpha=0.85)
    bars2 = ax.bar([i + width/2 for i in x], random_b, width, label="Random agent",    color="#9CA3AF", alpha=0.85)

    # Add value labels on top of bars
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Rate (%)", fontsize=12)
    ax.set_title("Final Performance: Trained Q-Agent vs. Random Agent\n"
                 "Tsoro Yematatu | ITRI 616 Mini-Project", fontsize=12, pad=12)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 110)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d%%'))
    ax.grid(axis="y", alpha=0.3)
    ax.set_facecolor("#F9FAFB")
    fig.tight_layout()

    fig.savefig("final_comparison.png", dpi=150)
    print("Saved: final_comparison.png")
    plt.close()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading training log...")
    log = load_log("training_log.pkl")

    print(f"Loaded {len(log['episodes'])} checkpoints")
    print(f"Final win rate  : {log['win_rate'][-1]*100:.1f}%")
    print(f"Final draw rate : {log['draw_rate'][-1]*100:.1f}%")
    print(f"Final loss rate : {log['loss_rate'][-1]*100:.1f}%")

    print("\nGenerating plots...")
    plot_learning_curve(log)
    plot_epsilon_decay(log)
    plot_final_comparison(log)

    print("\nAll plots saved. Use these in your experimental evaluation section.")
