# Snake Game with Subtle Fading Glow Effect

This project contains two versions of the classic Snake game implemented in Python using Pygame:

## Files
- `snake_game.py` - Original Snake game
- `snake_game_glow.py` - Enhanced version with subtle fading glow effect

## Features of the Glow Version
- **Subtle fading rectangular glow** - Soft, professional glow effect around snake and food
- **Pure rectangular shapes** - No curved corners, maintains classic Snake aesthetics
- **Larger 800x600 window** - Better visual experience and gameplay area
- **Improved collision detection** - Food properly aligns with snake movement grid
- **Soft color palette** - Light green snake glow, light red/pink food glow
- **Centered game over message** - Improved user interface
- **Virtual environment ready** - Dependencies managed in `venv/`

## How to Play
1. Navigate to the snake-game directory:
   ```bash
   cd /home/manvendras/claude-test/snake-game
   ```

2. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

3. Play the glowing version:
   ```bash
   python3 snake_game_glow.py
   ```

   Or play the original version:
   ```bash
   python3 snake_game.py
   ```

## Game Controls
- **Arrow Keys** - Move the snake in four directions
- **Eat the glowing food** - Grow longer and increase your score
- **Avoid collisions** - Don't hit the walls or your own tail
- **Game Over** - Press 'Q' to quit or 'C' to play again

## Requirements
- Python 3.x
- Pygame 2.0+ (already installed in the virtual environment)

## Notes
The ALSA audio errors shown on startup are harmless and related to the WSL2 environment - the game functions perfectly without audio.