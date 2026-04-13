import pygame
import random
from collections import deque
from nn import snake_NN, build_state
import torch
import matplotlib.pyplot as plt

## Display constants
YELLOW = (255, 255, 102)
BLACK  = (0, 0, 0)
RED    = (213, 50, 80)
GREEN  = (0, 255, 0)

DIS_WIDTH   = 600
DIS_HEIGHT  = 400
SNAKE_BLOCK = 10
SNAKE_SPEED = 300

## Training hyperparameters
BATCH_SIZE = 128
BUFFER_SIZE = 100_000
LR = 0.02
HIDDEN_DIM = 512

EPSILON_MIN   = 0.1
EPSILON_DECAY = 0.995
EXPLORATION_STEPS = 200_000

## Reward shaping
DEATH_PENALTY   = -50.0
STEP_PENALTY    = -0.1
DISTANCE_REWARD = 0.2
FOOD_REWARD     = 70.0

## Starvation limit: if the snake goes this many steps without eating, it dies
MAX_STEPS_WITHOUT_FOOD = 200

PLOT_REFRESH_INTERVAL = 50

## Direction helpers
DIRECTION_MAP  = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
DIR_ONE_HOT    = {"UP": [1,0,0,0], "DOWN": [0,1,0,0], "LEFT": [0,0,1,0], "RIGHT": [0,0,0,1]}
VELOCITY_TO_DIR = {
    (-SNAKE_BLOCK, 0): "LEFT",
    ( SNAKE_BLOCK, 0): "RIGHT",
    (0, -SNAKE_BLOCK): "UP",
    (0,  SNAKE_BLOCK): "DOWN",
}

## Pygame init
pygame.init()
dis = pygame.display.set_mode((DIS_WIDTH, DIS_HEIGHT))
pygame.display.set_caption('Snake Game')
clock = pygame.time.Clock()
score_font = pygame.font.SysFont("comicsansms", 35)

## Device & model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model     = snake_NN(hidden_dim=HIDDEN_DIM).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

## Replay buffer & training state
replay_buffer = deque(maxlen=BUFFER_SIZE)
loss_history  = []
TOTAL_STEPS   = 0
EPSILON       = 1.0

## Live loss plot
plt.ion()
fig, ax = plt.subplots(figsize=(6, 3))
ax.set_title("Training Loss")
ax.set_xlabel("Step")
ax.set_ylabel("Loss")
loss_line, = ax.plot([], [], color="cyan", linewidth=0.8)
fig.tight_layout()


def display_score(score):
    value = score_font.render("Score: " + str(score), True, YELLOW)
    dis.blit(value, [0, 0])


def draw_snake(snake_block, snake_list):
    for x in snake_list:
        pygame.draw.rect(dis, GREEN, [x[0], x[1], snake_block, snake_block])


def update_plot():
    if len(loss_history) < 2:
        return
    loss_line.set_data(range(len(loss_history)), loss_history)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()


def generate_food(snake_list):
    while True:
        foodx = round(random.randrange(0, DIS_WIDTH  - SNAKE_BLOCK) / 10.0) * 10.0
        foody = round(random.randrange(0, DIS_HEIGHT - SNAKE_BLOCK) / 10.0) * 10.0
        if not any(s[0] == foodx and s[1] == foody for s in snake_list):
            return foodx, foody


def train_step():
    batch = random.sample(replay_buffer, BATCH_SIZE)
    states, actions, rewards = zip(*batch)

    states  = torch.stack(states)
    actions = torch.tensor(actions, device=device)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)

    probs     = model.net(states)
    # Clamp to avoid log(0) → -inf which causes NaN loss when model is overconfident
    log_probs = torch.log(probs.clamp(min=1e-8))
    chosen    = log_probs[torch.arange(BATCH_SIZE), actions]

    loss = -(chosen * rewards).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())


def reset_episode():
    direction = DIRECTION_MAP[random.randint(0, 3)]
    return {
        "x1": DIS_WIDTH / 2,
        "y1": DIS_HEIGHT / 2,
        "x1_change": 0,
        "y1_change": 0,
        "snake_list": [],
        "length": 1,
        "direction":          direction,
        "current_direction":  DIR_ONE_HOT[direction],
        "action_idx":         0,
        "steps_since_food":   0,
    }


def die(s, foodx, foody, x1, y1):
    """Record a death transition and return a fresh episode state + new food."""
    global EPSILON
    dead_state = build_state(
        s["snake_list"], foodx, foody, x1, y1,
        s["current_direction"], s["steps_since_food"], device
    )
    replay_buffer.append((dead_state.detach(), s["action_idx"], DEATH_PENALTY))
    EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
    new_s = reset_episode()
    return new_s, generate_food(new_s["snake_list"])


def game_loop():
    global TOTAL_STEPS, EPSILON

    s = reset_episode()
    foodx, foody = generate_food(s["snake_list"])
    game_over = False

    while not game_over:
        ## Move snake; save prev position for distance-reward calculation
        prev_x, prev_y = s["x1"], s["y1"]
        s["x1"] += s["x1_change"]
        s["y1"] += s["y1_change"]
        x1, y1 = s["x1"], s["y1"]

        ## Wall collision
        if x1 >= DIS_WIDTH or x1 < 0 or y1 >= DIS_HEIGHT or y1 < 0:
            s, (foodx, foody) = die(s, foodx, foody, x1, y1)
            continue

        dis.fill(BLACK)
        pygame.draw.rect(dis, RED, [foodx, foody, SNAKE_BLOCK, SNAKE_BLOCK])

        snake_head = [x1, y1]
        s["snake_list"].append(snake_head)
        if len(s["snake_list"]) > s["length"]:
            del s["snake_list"][0]

        ## Self-collision
        if any(seg == snake_head for seg in s["snake_list"][:-1]):
            s, (foodx, foody) = die(s, foodx, foody, x1, y1)
            continue

        ## Starvation: too many steps without eating
        if s["steps_since_food"] >= MAX_STEPS_WITHOUT_FOOD:
            s, (foodx, foody) = die(s, foodx, foody, x1, y1)
            continue

        ## Build state once — used for both inference and replay buffer
        state = build_state(
            s["snake_list"], foodx, foody, x1, y1,
            s["current_direction"], s["steps_since_food"], device
        )
        output = model(state)

        ## Epsilon-greedy action selection
        if random.random() < EPSILON and TOTAL_STEPS < EXPLORATION_STEPS:
            s["action_idx"] = random.randint(0, 3)
        else:
            s["action_idx"] = torch.argmax(output).item()

        ## Apply velocity (no 180-degree reversal)
        requested = DIRECTION_MAP[s["action_idx"]]
        if requested == "LEFT"  and s["x1_change"] != SNAKE_BLOCK:
            s["x1_change"], s["y1_change"] = -SNAKE_BLOCK, 0
        elif requested == "RIGHT" and s["x1_change"] != -SNAKE_BLOCK:
            s["x1_change"], s["y1_change"] =  SNAKE_BLOCK, 0
        elif requested == "UP"    and s["y1_change"] != SNAKE_BLOCK:
            s["x1_change"], s["y1_change"] = 0, -SNAKE_BLOCK
        elif requested == "DOWN"  and s["y1_change"] != -SNAKE_BLOCK:
            s["x1_change"], s["y1_change"] = 0,  SNAKE_BLOCK

        ## Sync direction to *actual* movement so next state reflects reality
        actual = VELOCITY_TO_DIR.get((s["x1_change"], s["y1_change"]))
        if actual:
            s["direction"]         = actual
            s["current_direction"] = DIR_ONE_HOT[actual]

        draw_snake(SNAKE_BLOCK, s["snake_list"])
        display_score(s["length"] - 1)
        pygame.display.update()

        ## Per-step reward (freshly computed each step, never accumulated)
        reward = STEP_PENALTY
        prev_dist = abs(prev_x - foodx) + abs(prev_y - foody)
        curr_dist = abs(x1   - foodx) + abs(y1   - foody)
        if curr_dist < prev_dist:
            reward += DISTANCE_REWARD

        ## Food eaten
        s["steps_since_food"] += 1
        if x1 == foodx and y1 == foody:
            foodx, foody = generate_food(s["snake_list"])
            s["length"] += 1
            s["steps_since_food"] = 0
            reward += FOOD_REWARD

        replay_buffer.append((state.detach(), s["action_idx"], reward))

        ## Train when buffer is ready
        if len(replay_buffer) >= BATCH_SIZE:
            train_step()
            if TOTAL_STEPS % PLOT_REFRESH_INTERVAL == 0:
                update_plot()

        TOTAL_STEPS += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True

        clock.tick(SNAKE_SPEED)

    pygame.quit()
    plt.ioff()
    plt.show()
    quit()


game_loop()
