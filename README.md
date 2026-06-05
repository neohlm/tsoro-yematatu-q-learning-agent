# Tsoro Yematatu — Q-Learning Agent
### Mini-Project | Neo Hlumbene

> A tabular Q-Learning agent that learns to play Tsoro Yematatu, a traditional 
> two-player strategy game played by local South Africans and originally from Zimbabwe, entirely through self-play simulation.

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

## Requirements

- Python 3.7 or higher
- matplotlib (for graph generation only)

Check your Python version:
```bash
python --version
```

---

## Installation

### Step 1 — Clone the repository
```bash
git clone https://github.com/neohlm/tsoro-yematatu-q-learning-agent.git
cd tsoro-yematatu-q-learning-agent
```

### Step 2 — Install dependencies
```bash
pip install matplotlib
```

No other libraries are needed — the game engine and agent are pure Python.

---

## How to Run

### 1. Train the agent
```bash
python train.py
```

Runs 20,000 self-play games and evaluates the agent every 500 episodes 
against a random opponent. Training takes roughly 2–4 minutes on a standard laptop.

Output files created:
- `q_table.pkl` — the learned Q-table (saved agent)
- `training_log.pkl` — win/draw/loss rates recorded at each checkpoint

Example console output:
```
Episode    500 | Win: 54.5% | Draw: 3.0% | Loss: 42.5% | ε=0.779 | Q-entries: 312
Episode  1,000 | Win: 61.0% | Draw: 4.5% | Loss: 34.5% | ε=0.607 | Q-entries: 589
Episode  5,000 | Win: 78.5% | Draw: 6.0% | Loss: 15.5% | ε=0.082 | Q-entries: 1243
Episode 10,000 | Win: 85.0% | Draw: 5.0% | Loss: 10.0% | ε=0.050 | Q-entries: 1836
Episode 20,000 | Win: 88.0% | Draw: 4.5% | Loss:  7.5% | ε=0.050 | Q-entries: 1937
```

### 2. Generate graphs
```bash
python plot_results.py
```

Reads `training_log.pkl` and produces three PNG graph files:

| File | Description |
|---|---|
| `learning_curve.png` | Win/draw/loss rates across all training episodes |
| `final_comparison.png` | Trained agent vs. random agent bar chart |
| `epsilon_decay.png` | Exploration rate decay over training |

> **Note:** Run `train.py` before `plot_results.py` — the graphs depend on 
> the training log produced by training.

### 3. Play against the agent
```bash
python play.py
```

Loads the saved Q-table and lets you play as Player 1 (X) against the 
trained agent (O) in the terminal. Run `train.py` first to generate `q_table.pkl`.

---

## Running Experiments

To experiment with different hyperparameter settings, open `train.py` and 
edit the values at the top of the file:

```python
TOTAL_EPISODES   = 20_000   # increase for longer training
EVAL_INTERVAL    = 500      # how often to record performance
EVAL_GAMES       = 200      # games per evaluation checkpoint

ALPHA            = 0.1      # learning rate (try 0.01, 0.5)
GAMMA            = 0.95     # discount factor (try 0.8, 0.99)
EPSILON_START    = 1.0      # initial exploration rate
EPSILON_MIN      = 0.05     # minimum exploration rate
EPSILON_DECAY    = 0.9995   # decay rate per episode (try 0.999, 0.9999)
```

After changing any values, re-run the full pipeline:

```bash
python train.py
python plot_results.py
```

New graph files will overwrite the previous ones. To compare runs, rename 
your output files before re-running, for example:
```bash
# After first run
mv learning_curve.png learning_curve_alpha01.png

# Change alpha in train.py, then re-run
python train.py
python plot_results.py
mv learning_curve.png learning_curve_alpha05.png
```

---

## Board Layout

```
      0          ← top apex
     /|\
    / | \
   1--2--3       ← middle row (left, centre, right)
  /   |   \
 4----5----6     ← bottom row (left, centre, right)
```

Pieces are represented as:
- `.` — empty position
- `X` — Player 1
- `O` — Player 2 (agent)

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
| `gamma` | 0.95 | Discount factor — how much future rewards are valued |
| `epsilon_start` | 1.0 | Start fully exploring (random moves) |
| `epsilon_min` | 0.05 | Never go below 5% exploration |
| `epsilon_decay` | 0.9995 | Multiply epsilon by this value each episode |
| `total_episodes` | 20,000 | Total self-play training games |

---

## References
- Mitchell, T. M. (1997). *Machine Learning*. McGraw-Hill. (Chapter 1 — TEP framework)
- Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. *Machine Learning*, 8(3–4), 279–292.
- Wikipedia: [Tsoro Yematatu](https://en.wikipedia.org/wiki/Tsoro_Yematatu)