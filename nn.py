import torch
import torch.nn as nn


class snake_NN(nn.Module):
    GRID_COLS = 60   # DIS_WIDTH  / SNAKE_BLOCK = 600 / 10
    GRID_ROWS = 40   # DIS_HEIGHT / SNAKE_BLOCK = 400 / 10
    GRID_SIZE = GRID_COLS * GRID_ROWS   # 2400
    INPUT_DIM = GRID_SIZE + 2 + 2 + 4  # 2408

    def __init__(self, hidden_dim=256, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(self.INPUT_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
            nn.Softmax(dim=-1)
        )

    def build_state(self, snake_list, food_pos, head, current_direction):
        # Infer device from model weights — all tensors built here match it
        dev = next(self.parameters()).device

        # Grid: built without gradients (it's just a lookup, not a learnable op)
        with torch.no_grad():
            grid = torch.zeros(self.GRID_ROWS, self.GRID_COLS, device=dev)
            for sx, sy in snake_list:
                x = int(sx // 10)
                y = int(sy // 10)
                if 0 <= x < self.GRID_COLS and 0 <= y < self.GRID_ROWS:
                    grid[y][x] = 1

        # Normalize positions to [0, 1]
        norm = torch.tensor([600.0, 400.0], device=dev)
        food_norm = food_pos.float().to(dev) / norm
        head_norm  = head.float().to(dev)    / norm

        direction = torch.tensor(current_direction, dtype=torch.float32, device=dev)

        return torch.cat((grid.flatten(), food_norm, head_norm, direction))

    def forward(self, snake_list, food_pos, head, current_direction):
        state = self.build_state(snake_list, food_pos, head, current_direction)
        return self.net(state)