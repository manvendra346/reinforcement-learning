# Snake Game with Reinforcement Learning

A self-playing Snake game that trains a neural network agent using REINFORCE (policy gradient).

## Files
- `snake_game.py` - Game loop, training logic, and rendering
- `nn.py` - Neural network definition and state encoder
- `requirements.txt` - Python dependencies

## How it works

The agent observes a compact **11-dimensional** state vector:
- **4 values** — danger signals (wall or body 1 step away in UP/DOWN/LEFT/RIGHT)
- **2 values** — relative food direction (normalized dx, dy)
- **4 values** — current direction one-hot (UP / DOWN / LEFT / RIGHT)
- **1 value** — urgency (steps since last food eaten, normalized)

This is fed through a 2-hidden-layer network (11 → 128 → 64 → 4) that outputs a probability distribution over 4 actions. The agent **samples** actions from this distribution during training.

**Training algorithm — REINFORCE:**
- Collect a full episode trajectory (state, action, reward at each step)
- At episode end, compute **discounted returns** G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...
- Normalize returns (zero mean, unit std) for variance reduction
- Update policy: reinforce actions that led to higher returns
- Entropy bonus prevents premature policy collapse
- Gradient clipping for training stability

**Reward shaping:**
- `-0.01` per step (discourages wandering)
- `+0.02` when moving closer to food
- `+1.0` for eating food
- `-1.0` for dying (wall collision, self-collision, or starvation)
- Starvation: agent dies after 200 steps without eating

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
2. **Loss plot** — live training loss updates every 50 episodes

## What to watch for as it learns

| Phase | What you'll see |
|---|---|
| **Early (0–200 episodes)** | Snake moves somewhat randomly, dies quickly, explores via sampling |
| **Mid (200–1000 episodes)** | Snake starts heading toward food more consistently, survives longer |
| **Late (1000+ episodes)** | Snake navigates toward food reliably, avoids walls and its own body |

The loss plot will appear noisy at first — that's normal. A downward trend over hundreds of episodes is the signal that learning is happening.

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
