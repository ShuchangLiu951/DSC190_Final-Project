import pygame
import random
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from collections import namedtuple

# ------------- GAME CONSTANTS -------------

BLOCK_SIZE = 20
SPEED = 80  # FPS when training/playing

WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

WIDTH = 640
HEIGHT = 480

pygame.init()
font = pygame.font.SysFont('arial', 25)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x y')


# ------------- SNAKE GAME ENVIRONMENT -------------

class SnakeGameAI:
    def __init__(self, w=WIDTH, h=HEIGHT, render=True):
        self.w = w
        self.h = h
        self.render = render

        if self.render:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('RL Snake')
            self.clock = pygame.time.Clock()
        else:
            self.display = None
            self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w // 2, self.h // 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - 2 * BLOCK_SIZE, self.head.y)
        ]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    @staticmethod
    def _distance(p1, p2):
        # Manhattan distance
        return abs(p1.x - p2.x) + abs(p1.y - p2.y)

    def play_step(self, action):
        """
        action: [1,0,0] -> straight, [0,1,0] -> right turn, [0,0,1] -> left turn
        returns: reward, done, score
        """
        self.frame_iteration += 1

        # 1. handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # distance to food BEFORE moving (for reward shaping)
        old_distance = self._distance(self.head, self.food)

        # 2. move
        self._move(action)
        self.snake.insert(0, self.head)

        reward = 0
        done = False

        # 3. check collision / timeout
        if self.is_collision() or self.frame_iteration > 150 * len(self.snake):
            done = True
            reward = -15  # punishment for dying or stalling
            return reward, done, self.score

        # 4. check food
        if self.head == self.food:
            self.score += 1
            reward = 20  # strong reward for eating
            self._place_food()
        else:
            # reward shaping based on distance to food
            new_distance = self._distance(self.head, self.food)
            if new_distance < old_distance:
                reward = 1.0   # moved closer
            else:
                reward = -1.0  # moved away or same
            self.snake.pop()

        # 5. update ui and clock
        if self.render:
            self._update_ui()
            self.clock.tick(SPEED)

        return reward, done, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head

        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True

        # hits itself
        if pt in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        # current direction
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = clock_wise[(idx + 1) % 4]  # right turn
        else:  # [0,0,1]
            new_dir = clock_wise[(idx - 1) % 4]  # left turn

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)

    # --- STATE REPRESENTATION FOR RL ---

    def get_state(self):
        head = self.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and self.is_collision(point_r)) or
            (dir_l and self.is_collision(point_l)) or
            (dir_u and self.is_collision(point_u)) or
            (dir_d and self.is_collision(point_d)),

            # Danger right
            (dir_u and self.is_collision(point_r)) or
            (dir_d and self.is_collision(point_l)) or
            (dir_l and self.is_collision(point_u)) or
            (dir_r and self.is_collision(point_d)),

            # Danger left
            (dir_d and self.is_collision(point_r)) or
            (dir_u and self.is_collision(point_l)) or
            (dir_r and self.is_collision(point_u)) or
            (dir_l and self.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            self.food.x < head.x,  # food left
            self.food.x > head.x,  # food right
            self.food.y < head.y,  # food up
            self.food.y > head.y   # food down
        ]

        return np.array(state, dtype=int)


# ------------- Q-LEARNING AGENT -------------

class Agent:
    def __init__(self):
        self.gamma = 0.95     # discount factor
        self.lr = 0.1

        # epsilon-greedy params
        self.epsilon = 1.0
        self.eps_min = 0.05
        self.eps_decay_factor = 0.98  # per game

        self.n_actions = 3   # [straight, right, left]
        self.Q = {}          # dict: state tuple -> q-values array

        self.games_played = 0
        self.record = 0

    def get_state_key(self, state):
        return tuple(state.tolist())

    def get_q_values(self, state_key):
        if state_key not in self.Q:
            self.Q[state_key] = np.zeros(self.n_actions, dtype=float)
        return self.Q[state_key]

    def choose_action(self, state):
        state_key = self.get_state_key(state)
        q_values = self.get_q_values(state_key)

        # epsilon-greedy policy
        if random.random() < self.epsilon:
            action_idx = random.randint(0, self.n_actions - 1)
        else:
            action_idx = int(np.argmax(q_values))

        action = np.zeros(self.n_actions, dtype=int)
        action[action_idx] = 1
        return action, action_idx

    def learn(self, state, action_idx, reward, next_state, done):
        s_key = self.get_state_key(state)
        ns_key = self.get_state_key(next_state)

        q_values = self.get_q_values(s_key)
        next_q_values = self.get_q_values(ns_key)

        q_old = q_values[action_idx]
        q_target = reward + (0 if done else self.gamma * np.max(next_q_values))
        q_values[action_idx] = q_old + self.lr * (q_target - q_old)

    def decay_epsilon(self):
        if self.epsilon > self.eps_min:
            self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay_factor)


# ------------- TRAINING LOOP -------------

def train(render=True, max_games=400):
    game = SnakeGameAI(render=render)
    agent = Agent()

    scores = []
    mean_scores = []

    while agent.games_played < max_games:
        state_old = game.get_state()
        action, action_idx = agent.choose_action(state_old)
        reward, done, score = game.play_step(action)
        state_new = game.get_state()

        agent.learn(state_old, action_idx, reward, state_new, done)

        if done:
            agent.games_played += 1
            game.reset()

            # stats
            scores.append(score)
            mean_scores.append(sum(scores) / len(scores))

            if score > agent.record:
                agent.record = score

            agent.decay_epsilon()

            print(
                f'Game {agent.games_played:3d}  '
                f'Score: {score:2d}  Record: {agent.record:2d}  '
                f'Epsilon: {agent.epsilon:.3f}'
            )

    print("Training finished!")
    return agent, scores, mean_scores


# ------------- PLOTTING -------------

def plot_training(scores, mean_scores):
    plt.figure(figsize=(10, 5))
    plt.plot(scores, label="Score per Game")
    plt.plot(mean_scores, label="Mean Score", linewidth=3)
    plt.xlabel("Game")
    plt.ylabel("Score")
    plt.title("Training Curve: Score vs Game")
    plt.legend()
    plt.grid(True)
    plt.show()


# ------------- EVALUATION MODE -------------

def evaluate_agent(agent, n_games=50):
    """Runs N games with epsilon=0 and reports average performance."""
    agent.epsilon = 0.0   # greedy policy
    game = SnakeGameAI(render=False)

    scores = []

    for _ in range(n_games):
        game.reset()
        done = False

        while not done:
            state = game.get_state()
            key = agent.get_state_key(state)
            q_values = agent.get_q_values(key)
            action_idx = int(np.argmax(q_values))

            action = np.zeros(agent.n_actions, dtype=int)
            action[action_idx] = 1

            reward, done, score = game.play_step(action)

        scores.append(score)

    avg_score = sum(scores) / len(scores)
    max_score = max(scores)
    min_score = min(scores)

    print(f"\nEvaluation over {n_games} games:")
    print(f"  Average score: {avg_score:.2f}")
    print(f"  Max score:     {max_score}")
    print(f"  Min score:     {min_score}")

    return scores


# ------------- PLAY WITH TRAINED AGENT -------------

def watch_trained_agent(agent):
    game = SnakeGameAI(render=True)
    game.reset()
    agent.epsilon = 0.0  # greedy play during demo

    while True:
        state = game.get_state()
        state_key = agent.get_state_key(state)
        q_values = agent.get_q_values(state_key)
        a_idx = int(np.argmax(q_values))
        action = np.zeros(agent.n_actions, dtype=int)
        action[a_idx] = 1

        reward, done, score = game.play_step(action)

        if done:
            print("Episode finished. Score:", score)
            game.reset()


def plot_evaluation_boxplot(eval_scores):
    """Creates a simple boxplot of evaluation performance."""
    plt.figure(figsize=(6, 5))
    plt.boxplot(eval_scores, vert=True, patch_artist=True)

    plt.title("Evaluation Score Distribution After Training")
    plt.ylabel("Score")

    # Visual improvements
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    plt.show()

# ------------- MAIN -------------

if __name__ == '__main__':
    # 1) Train the agent (no rendering = much faster).
    trained_agent, scores, mean_scores = train(render=False, max_games=500)

    # 2) Plot training curve.
    plot_training(scores, mean_scores)

    # 3) Evaluate trained agent.
    eval_scores = evaluate_agent(trained_agent, n_games=50)

    # 3b) Plot boxplot of evaluation results
    plot_evaluation_boxplot(eval_scores)

    # 4) Watch the agent play.
    watch_trained_agent(trained_agent)