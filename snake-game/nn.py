import torch
import torch.nn as nn

GRID_COLS = 60   # DIS_WIDTH  / SNAKE_BLOCK = 600 / 10
GRID_ROWS = 40   # DIS_HEIGHT / SNAKE_BLOCK = 400 / 10
#                  grid       + food_norm + head_norm + direction + steps_since_food
INPUT_DIM = GRID_COLS * GRID_ROWS + 2 + 2 + 4 + 1  # 2409


def build_state(snake_list, foodx, foody, head_x, head_y, current_direction, steps_since_food, device):
    """Encode raw game values into a normalized input tensor of shape (INPUT_DIM,).

    Args:
        snake_list: List of [x, y] pixel coordinates (head to tail)
        foodx, foody: Food position in pixels
        head_x, head_y: Snake head position in pixels
        current_direction: One-hot list [UP, DOWN, LEFT, RIGHT]
        steps_since_food: Steps elapsed since the last food was eaten (urgency signal)
        device: torch device

    Returns:
        Tensor of shape (INPUT_DIM,)
    """
    with torch.no_grad():
        grid = torch.zeros(GRID_ROWS, GRID_COLS, device=device)
        for sx, sy in snake_list:
            x, y = int(sx // 10), int(sy // 10)
            if 0 <= x < GRID_COLS and 0 <= y < GRID_ROWS:
                grid[y][x] = 1

    food_norm  = torch.tensor([foodx / 600.0, foody / 400.0], dtype=torch.float32, device=device)
    head_norm  = torch.tensor([head_x / 600.0, head_y / 400.0], dtype=torch.float32, device=device)
    direction  = torch.tensor(current_direction, dtype=torch.float32, device=device)
    urgency    = torch.tensor([steps_since_food / 500.0], dtype=torch.float32, device=device)
    return torch.cat((grid.flatten(), food_norm, head_norm, direction, urgency))


class snake_NN(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        """Forward pass on a pre-built state tensor."""
        return self.net(state)
