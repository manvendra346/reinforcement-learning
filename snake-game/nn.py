import torch
import torch.nn as nn

INPUT_DIM = 11  # 4 danger + 2 food_dir + 4 direction + 1 urgency


def build_state(snake_list, foodx, foody, head_x, head_y,
                current_direction, steps_since_food, device,
                dis_width=600, dis_height=400, snake_block=10):
    """Encode game state into a compact 11-dim tensor.

    Features:
        [0-3] danger signals — wall or body 1 step away (UP, DOWN, LEFT, RIGHT)
        [4-5] relative food direction — normalized dx, dy
        [6-9] current direction one-hot (UP, DOWN, LEFT, RIGHT)
        [10]  urgency — steps since last food, normalized
    """
    body = snake_list[:-1] if len(snake_list) > 1 else []

    ## Are these danger signals even useful?
    ## This incorporates learning where head is indirectly
    danger_up = float(
        head_y - snake_block < 0
        or [head_x, head_y - snake_block] in body
    )
    danger_down = float(
        head_y + snake_block >= dis_height
        or [head_x, head_y + snake_block] in body
    )
    danger_left = float(
        head_x - snake_block < 0
        or [head_x - snake_block, head_y] in body
    )
    danger_right = float(
        head_x + snake_block >= dis_width
        or [head_x + snake_block, head_y] in body
    )

    food_dx = (foodx - head_x) / dis_width
    food_dy = (foody - head_y) / dis_height

    urgency = steps_since_food / 300.0

    return torch.tensor(
        [danger_up, danger_down, danger_left, danger_right,
         food_dx, food_dy] + current_direction + [urgency],
        dtype=torch.float32, device=device
    )


class snake_NN(nn.Module):
    def __init__(self, hidden1=128, hidden2=64, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, num_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        """Forward pass on a pre-built state tensor."""
        return self.net(state)
