# Snake Game with Reinforcement Learning

A self-playing Snake game that trains a neural network agent using policy gradient RL.

## Files
- `snake_game.py` - Game loop, training logic, and rendering
- `nn.py` - Neural network definition and state encoder
- `requirements.txt` - Python dependencies

## How it works

The agent observes a 2408-dimensional state vector:
- **2400 values** — 60×40 grid with 1s where the snake body is
- **2 values** — normalized food position
- **2 values** — normalized head position
- **4 values** — one-hot direction (UP / DOWN / LEFT / RIGHT)

This is fed through a 2-layer feed-forward network that outputs a probability over 4 actions. The agent trains with a simple policy gradient: actions that led to higher rewards are reinforced.

**Reward shaping:**
- `-0.4` per step (discourages wandering)
- `+2.0` when moving closer to food
- `+20.0` for eating food
- `-30.0` for dying (wall or self-collision)

Exploration uses epsilon-greedy decay from 1.0 → 0.1 over 100k steps.

## Setup

```bash
cd snake-game

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Running the game

```bash
python snake_game.py
```

Two windows will open side by side:
1. **Game window** — watch the snake move and grow in real time
2. **Loss plot** — live training loss updates every 50 steps

## What to watch for as it learns

| Phase | What you'll see |
|---|---|
| **Early (0–5k steps)** | Snake moves randomly, dies quickly, mostly explores |
| **Mid (5k–50k steps)** | Epsilon decays, snake starts heading toward food more consistently |
| **Late (50k+ steps)** | Exploitation dominates, snake navigates toward food reliably; loss curve flattens |

The loss plot will appear noisy at first — that's normal. A downward trend over thousands of steps is the signal that learning is happening.

Close the game window to stop training and display the final loss curve.

## Requirements
- Python 3.x
- PyTorch >= 2.10.0
- Pygame >= 2.6.1
- Matplotlib >= 3.10.8
- NumPy >= 2.3.5

## Notes
- ALSA audio warnings on startup are harmless (WSL2 environment)
- GPU is used automatically if available (`cuda`), otherwise falls back to CPU
