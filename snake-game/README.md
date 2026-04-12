# Snake Game with Reinforcement Learning

A self-playing Snake game that trains a neural network agent using policy gradient RL.

## Files
- `snake_game.py` - Game loop, training logic, and rendering
- `nn.py` - Neural network definition and state encoder

## How it works

The agent observes a 2408-dimensional state vector:
- **2400 values** — 60×40 grid with 1s where the snake body is
- **2 values** — normalized food position
- **2 values** — normalized head position
- **4 values** — one-hot direction (UP / DOWN / LEFT / RIGHT)

This is fed through a 2-layer feed-forward network that outputs a probability over 4 actions. The agent is trained with a simple policy gradient: actions that led to higher rewards are reinforced.

**Reward shaping:**
- `-0.4` per step (discourages wandering)
- `+2.0` when moving closer to food
- `+20.0` for eating food
- `-30.0` for dying (wall or self-collision)

Exploration uses epsilon-greedy decay from 1.0 → 0.1 over 100k steps.

## Setup

```bash
cd snake-game
source venv/bin/activate
python snake_game.py
```

## Requirements
- Python 3.x
- PyTorch
- Pygame 2.0+
- Matplotlib

## Notes
- ALSA audio warnings on startup are harmless (WSL2 environment)
- A live training loss plot appears alongside the game window
- Close the game window to stop training and display the final loss curve
