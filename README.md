# Tsoro Yematatu — Q-Learning Agent
### ITRI 616 Mini-Project

---

## Project Structure

```
tsoro_yematatu/
├── game.py          # Game environment (board, rules, win detection)
├── agent.py         # Q-Learning agent + Random baseline agent
├── train.py         # Self-play training loop + evaluation
├── plot_results.py  # Generates graphs for your report
├── play.py          # Play against the trained agent in the terminal
└── README.md
```

---

## Setup

Install the only required library (matplotlib for graphs):

```bash
pip install matplotlib
```

No other libraries needed — pure Python.

---

## How to Run

### Step 1 — Train the agent
```bash
python train.py
```
This runs 20,000 self-play games and evaluates every 500 episodes.
Training takes roughly **1–3 minutes** on a standard laptop.

Output:
- `q_table.pkl` — the learned Q-table
- `training_log.pkl` — win/draw/loss rates per checkpoint

Example console output:
```
Episode  500 | Win: 54.5% | Draw: 3.0% | Loss: 42.5% | ε=0.779
Episode 1000 | Win: 61.0% | Draw: 4.5% | Loss: 34.5% | ε=0.607
Episode 5000 | Win: 78.5% | Draw: 6.0% | Loss: 15.5% | ε=0.082
Episode 10000| Win: 85.0% | Draw: 5.0% | Loss: 10.0% | ε=0.050
```

### Step 2 — Generate graphs
```bash
python plot_results.py
```
Produces three PNG files for your report:
- `learning_curve.png`   — win/draw/loss rates over training (main graph)
- `epsilon_decay.png`    — exploration rate decay over training
- `final_comparison.png` — trained agent vs. random agent bar chart

### Step 3 — Play against the agent (optional)
```bash
python play.py
```

---

## TEP Summary

| Component | Definition |
|---|---|
| **Task** | Play Tsoro Yematatu: place and move 3 pieces to form a 3-in-a-row |
| **Experience** | Self-play: thousands of simulated games with reward signals |
| **Performance** | Win rate (%) against a random opponent over 200 evaluation games |

---

## Hyperparameters

| Parameter | Value | Meaning |
|---|---|---|
| `alpha` | 0.1 | Learning rate — how fast Q-values update |
| `gamma` | 0.95 | Discount factor — value of future rewards |
| `epsilon_start` | 1.0 | Start fully exploring (random moves) |
| `epsilon_min` | 0.05 | Never go below 5% exploration |
| `epsilon_decay` | 0.9995 | Reduce epsilon by this factor each episode |
| `total_episodes` | 20,000 | Total self-play training games |

---

## References
- Mitchell, T. M. (1997). *Machine Learning*. McGraw-Hill. (Chapter 1 — TEP framework)
- Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. *Machine Learning*, 8(3–4), 279–292.
- Wikipedia: [Tsoro Yematatu](https://en.wikipedia.org/wiki/Tsoro_Yematatu)
