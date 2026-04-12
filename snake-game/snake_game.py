import pygame
import random
from collections import deque
from nn import snake_NN
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

## Variables
WHITE = (255, 255, 255)
YELLOW = (255, 255, 102)
BLACK = (0, 0, 0)
RED = (213, 50, 80)
GREEN = (0, 255, 0)

DIS_WIDTH = 600
DIS_HEIGHT = 400
SNAKE_BLOCK = 10
SNAKE_SPEED = 30

TOTAL_STEPS = 0

BATCH_SIZE = 256
BUFFER_SIZE = 10_000
replay_buffer = deque(maxlen=BUFFER_SIZE)

LR = 0.02
HIDDEN_DIM=256
# FIX 1: EPSILON moved to module level so it persists across episodes
EPSILON = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.99

DIRECTION_MAP = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT",
}

## INIT
pygame.init()
dis = pygame.display.set_mode((DIS_WIDTH, DIS_HEIGHT))
pygame.display.set_caption('Snake Game')
clock = pygame.time.Clock()
font_style = pygame.font.SysFont("bahnschrift", 25)
score_font = pygame.font.SysFont("comicsansms", 35)

## Helpers
def display_score(score):
    value = score_font.render("Your Score: " + str(score), True, YELLOW)
    dis.blit(value, [0, 0])

def draw_snake(snake_block, snake_list):
    for x in snake_list:
        pygame.draw.rect(dis, GREEN, [x[0], x[1], snake_block, snake_block])

def display_message(msg, color):
    mesg = font_style.render(msg, True, color)
    dis.blit(mesg, [DIS_WIDTH / 6, DIS_HEIGHT / 3])

## Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

## Model
model = snake_NN(hidden_dim=HIDDEN_DIM).to(device)
optim = torch.optim.Adam(model.parameters(), lr=LR)

## Live loss plot
loss_history = []
plt.ion()
fig, ax = plt.subplots(figsize=(6, 3))
ax.set_title("Training Loss")
ax.set_xlabel("Step")
ax.set_ylabel("Loss")
loss_line, = ax.plot([], [], color="cyan", linewidth=0.8)
fig.tight_layout()

def update_plot():
    if len(loss_history) < 2:
        return
    loss_line.set_data(range(len(loss_history)), loss_history)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()


def game_loop(enable_nn=False):
    global TOTAL_STEPS, EPSILON

    game_over = False
    game_close = False

    x1 = DIS_WIDTH / 2
    y1 = DIS_HEIGHT / 2
    x1_change = 0
    y1_change = 0

    snake_List = []
    length_of_snake = 1

    def generate_food(snake_list):
        while True:
            foodx = round(random.randrange(0, DIS_WIDTH - SNAKE_BLOCK) / 10.0) * 10.0
            foody = round(random.randrange(0, DIS_HEIGHT - SNAKE_BLOCK) / 10.0) * 10.0
            overlap = any(s[0] == foodx and s[1] == foody for s in snake_list)
            if not overlap:
                return foodx, foody

    ## Initial values
    foodx, foody = generate_food(snake_List)
    direction = DIRECTION_MAP[random.randint(0, 3)]
    current_direction = [1, 0, 0, 0]  # default: UP
    reward = 0
    while not game_over:

        ## Episode reset on death
        while game_close:
            if enable_nn:
                reward=0
                x1 = DIS_WIDTH / 2
                y1 = DIS_HEIGHT / 2
                x1_change = 0
                y1_change = 0
                snake_List = []
                length_of_snake = 1
                foodx, foody = generate_food(snake_List)
                direction = DIRECTION_MAP[random.randint(0, 3)]
                current_direction = [1, 0, 0, 0]
                game_close = False
                EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)

        ## Move snake 
        x1 += x1_change
        y1 += y1_change

        # When snake hits boundary, y=400 → index 40 = out of bounds for zeros(40,60)
        hit_wall = x1 >= DIS_WIDTH or x1 < 0 or y1 >= DIS_HEIGHT or y1 < 0
        if hit_wall:
            game_close = True
            dead_state = model.build_state(snake_List, torch.tensor([foodx, foody], device=device), torch.tensor([x1, y1], device=device), current_direction or [1,0,0,0])
            replay_buffer.append((dead_state.detach(), action_idx, -30.0))

        # Only run model and training if snake is alive
        if not game_close:
            dis.fill(BLACK)
            pygame.draw.rect(dis, RED, [foodx, foody, SNAKE_BLOCK, SNAKE_BLOCK])

            snake_Head = [x1, y1]
            snake_List.append(snake_Head)
            if len(snake_List) > length_of_snake:
                del snake_List[0]

            # Check self-collision
            for seg in snake_List[:-1]:
                if seg == snake_Head:
                    game_close = True
                    dead_state = model.build_state(snake_List, food_pos, head_pos, current_direction)
                    replay_buffer.append((dead_state.detach(), action_idx, -30.0))

            if not game_close:
                food_pos = torch.tensor([foodx, foody], device=device)
                head_pos = torch.tensor(snake_Head, device=device)

                # Build current_direction one-hot
                dir_map = {"UP": [1,0,0,0], "DOWN": [0,1,0,0],
                           "LEFT": [0,0,1,0], "RIGHT": [0,0,0,1]}
                current_direction = dir_map[direction]

                output = model(snake_List, food_pos, head_pos, current_direction)

                # FIX 4: Epsilon-greedy with correct decay direction
                if random.random() < EPSILON and TOTAL_STEPS < 100000:
                    action_idx = random.randint(0, 3)           # explore
                else:
                    action_idx = torch.argmax(output).item()    # exploit

                direction = DIRECTION_MAP[action_idx]

                # Handle quit event
                if enable_nn:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            game_over = True

                if direction == "LEFT" and x1_change != SNAKE_BLOCK:
                    x1_change, y1_change = -SNAKE_BLOCK, 0
                elif direction == "RIGHT" and x1_change != -SNAKE_BLOCK:
                    x1_change, y1_change = SNAKE_BLOCK, 0
                elif direction == "UP" and y1_change != SNAKE_BLOCK:
                    x1_change, y1_change = 0, -SNAKE_BLOCK
                elif direction == "DOWN" and y1_change != -SNAKE_BLOCK:
                    x1_change, y1_change = 0, SNAKE_BLOCK

                draw_snake(SNAKE_BLOCK, snake_List)
                display_score(length_of_snake - 1)
                pygame.display.update()

                # Compute reward
                reward += -0.4  # small step penalty to discourage wandering

                # Distance reward
                prev_dist = abs(x1 - x1_change - foodx) + abs(y1 - y1_change - foody)
                curr_dist = abs(x1 - foodx) + abs(y1 - foody)
                if curr_dist < prev_dist: reward += 2.0

                if x1 == foodx and y1 == foody:
                    foodx, foody = generate_food(snake_List)
                    length_of_snake += 1
                    reward += 20

                # Store this step in the replay buffer
                state = model.build_state(snake_List, food_pos, head_pos, current_direction)
                replay_buffer.append((state.detach(), action_idx, reward))

                # Train only once buffer has enough samples
                if len(replay_buffer) >= BATCH_SIZE:
                    batch = random.sample(replay_buffer, BATCH_SIZE)
                    states, actions, rewards = zip(*batch)

                    states  = torch.stack(states)                                        # (64, 2408)
                    actions = torch.tensor(actions, device=device)                       # (64,)
                    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)  # (64,)

                    probs     = model.net(states)                                        # (64, 4)
                    log_probs = torch.log(probs)                                         # (64, 4)
                    chosen    = log_probs[torch.arange(BATCH_SIZE), actions]             # (64,)

                    loss = -(chosen * rewards).mean()
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                    loss_history.append(loss.item())
                    if TOTAL_STEPS % 50 == 0:   # refresh plot every 50 steps
                        update_plot()

                TOTAL_STEPS += 1
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                clock.tick(SNAKE_SPEED)

    pygame.quit()
    plt.ioff()
    plt.show()
    quit()


game_loop(enable_nn=True)