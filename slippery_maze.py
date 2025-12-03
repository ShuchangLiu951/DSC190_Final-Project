import numpy as np
import random
import matplotlib.pyplot as plt
import pygame

# ==========================
# Environment: Slippery Maze
# ==========================

class SlipperyMazeEnv:
    """
    Simple gridworld with:
        S - start
        G - goal (reward +1, terminal)
        H - hole (reward -1, terminal)
        # - wall (blocked)
        . - empty

    Actions: 0 = UP, 1 = RIGHT, 2 = DOWN, 3 = LEFT
    Slippage: with probability slip_prob, a random action is executed instead.
    """

    def __init__(self, slip_prob=0.2, max_steps=100):
        # Define the grid as a list of strings (rows)
        self.grid = [
            "S....H...",
            ".##.#.##.",
            "..#...#..",
            "..##H.#..",
            "H.##..#..",
            "...#.H##.",
            "H..#.....",
            "..##.##..",
            "H....#..G"
        ]

        self.n_rows = len(self.grid)
        self.n_cols = len(self.grid[0])
        self.slip_prob = slip_prob
        self.max_steps = max_steps

        # Locate start state
        self.start_state = self._find_char('S')
        self.state = self.start_state
        self.steps = 0

        # Actions: up, right, down, left
        self.action_space = [0, 1, 2, 3]
        self.n_actions = len(self.action_space)
        self.n_states = self.n_rows * self.n_cols  # each cell index

    def _find_char(self, ch):
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                if self.grid[r][c] == ch:
                    return self._pos_to_state((r, c))
        raise ValueError(f"Character {ch} not found in grid")

    def _state_to_pos(self, s):
        return divmod(s, self.n_cols)  # (row, col)

    def _pos_to_state(self, pos):
        r, c = pos
        return r * self.n_cols + c

    def _cell_type(self, s):
        r, c = self._state_to_pos(s)
        return self.grid[r][c]

    def reset(self):
        self.state = self.start_state
        self.steps = 0
        return self.state

    def step(self, action):
        """
        Take an action with slippage.
        Returns: next_state, reward, done, info
        """
        self.steps += 1

        # Apply slippage
        if random.random() < self.slip_prob:
            action = random.choice(self.action_space)

        r, c = self._state_to_pos(self.state)

        # Compute candidate next position
        if action == 0:   # UP
            nr, nc = r - 1, c
        elif action == 1: # RIGHT
            nr, nc = r, c + 1
        elif action == 2: # DOWN
            nr, nc = r + 1, c
        else:             # LEFT
            nr, nc = r, c - 1

        # Check boundaries and walls
        if 0 <= nr < self.n_rows and 0 <= nc < self.n_cols and self.grid[nr][nc] != '#':
            next_state = self._pos_to_state((nr, nc))
        else:
            # Invalid move: stay in place
            next_state = self.state

        cell = self._cell_type(next_state)

        # Rewards and terminal conditions
        if cell == 'G':
            reward = 1.0
            done = True
        elif cell == 'H':
            reward = -2.0
            done = True
        else:
            reward = -0.02  # small step penalty to encourage shorter paths
            done = False

        # Max steps safety
        if self.steps >= self.max_steps:
            done = True

        self.state = next_state
        return next_state, reward, done, {}

    def render_policy(self, Q):
        """
        Render greedy policy from Q-table as arrows.
        """
        arrow_map = {
            0: '↑',
            1: '→',
            2: '↓',
            3: '←'
        }
        print("Greedy policy (excluding walls):")
        for r in range(self.n_rows):
            row_str = ""
            for c in range(self.n_cols):
                s = self._pos_to_state((r, c))
                cell = self.grid[r][c]
                if cell == '#':
                    row_str += " # "
                elif cell in ['G', 'H', 'S']:
                    row_str += f" {cell} "
                else:
                    best_a = np.argmax(Q[s])
                    row_str += f" {arrow_map[best_a]} "
            print(row_str)
        print()

    def render_value(self, Q):
        """
        Show value function V(s) = max_a Q(s,a) as a matrix.
        """
        V = np.zeros((self.n_rows, self.n_cols))
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                s = self._pos_to_state((r, c))
                if self.grid[r][c] == '#':
                    V[r, c] = np.nan  # walls
                else:
                    V[r, c] = np.max(Q[s])
        print("State values (max_a Q(s,a)):")
        print(np.round(V, 2))
        print()


# ======================
# Q-learning Training
# ======================

def train_q_learning(
    env,
    num_episodes=700,
    alpha=0.1,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=0.997,
    snapshot_points=None    # <── NEW
):
    if snapshot_points is None:
        snapshot_points = []

    Q = np.zeros((env.n_states, env.n_actions))
    rewards_per_episode = []

    # Track successes (reaching the goal)
    success_counts = []  # success rate per 100 episodes
    episodes_window = 100
    success_in_window = 0

    epsilon = epsilon_start

    # NEW: store Q-table snapshots at selected episodes
    Q_snapshots = {}

    for episode in range(num_episodes):
        s = env.reset()
        done = False
        total_reward = 0.0
        reached_goal = False

        while not done:
            # ε-greedy action selection
            if random.random() < epsilon:
                a = random.choice(env.action_space)
            else:
                a = int(np.argmax(Q[s]))

            s_next, r, done, _ = env.step(a)
            total_reward += r

            # Track if we reached the goal
            if env._cell_type(s_next) == 'G':
                reached_goal = True

            # Q-learning update
            best_next = np.max(Q[s_next])
            Q[s, a] = Q[s, a] + alpha * (r + gamma * best_next - Q[s, a])

            s = s_next

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        rewards_per_episode.append(total_reward)

        # Update success counts
        if reached_goal:
            success_in_window += 1

        # Every 100 episodes, compute success rate
        if (episode + 1) % episodes_window == 0:
            success_rate = success_in_window / episodes_window
            success_counts.append(success_rate)
            print(f"Episode {episode+1}/{num_episodes}, "
                  f"Success rate (last 100): {success_rate:.2f}")
            success_in_window = 0  # reset counter

        # ⬅️ NEW: save snapshot if this is a requested episode
        if (episode + 1) in snapshot_points:
            Q_snapshots[episode + 1] = Q.copy()

    # Also store the final Q-table
    Q_snapshots[num_episodes] = Q.copy()

    return Q, rewards_per_episode, success_counts, Q_snapshots



# ======================
# SARSA
# ======================

def train_sarsa(
    env,
    num_episodes=1500,
    alpha=0.1,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=0.997
):
    Q = np.zeros((env.n_states, env.n_actions))
    rewards_per_episode = []
    success_rates = []
    episodes_window = 100
    success_in_window = 0

    epsilon = epsilon_start

    for episode in range(num_episodes):
        s = env.reset()
        done = False
        total_reward = 0.0

        # Choose initial action using ε-greedy
        if random.random() < epsilon:
            a = random.choice(env.action_space)
        else:
            a = int(np.argmax(Q[s]))

        reached_goal = False

        while not done:
            s_next, r, done, _ = env.step(a)
            total_reward += r

            if env._cell_type(s_next) == 'G':
                reached_goal = True

            # Choose next action A' using ε-greedy (SARSA)
            if not done:
                if random.random() < epsilon:
                    a_next = random.choice(env.action_space)
                else:
                    a_next = int(np.argmax(Q[s_next]))
            else:
                a_next = None

            # SARSA update: Q(s,a) <- Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]
            if not done:
                target = r + gamma * Q[s_next, a_next]
            else:
                target = r  # terminal state's Q = 0

            Q[s, a] += alpha * (target - Q[s, a])

            # Move forward
            s, a = s_next, a_next

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        rewards_per_episode.append(total_reward)

        # Track success
        if reached_goal:
            success_in_window += 1

        if (episode + 1) % episodes_window == 0:
            success_rate = success_in_window / episodes_window
            success_rates.append(success_rate)
            print(f"[SARSA] Episode {episode+1}/{num_episodes}, Success rate: {success_rate:.2f}")
            success_in_window = 0

    return Q, rewards_per_episode, success_rates


# ======================
# Evaluation
# ======================

def evaluate_policy(env, Q, num_episodes=20):
    """
    Run greedy policy (no exploration) and report average return.
    """
    total_returns = []
    for _ in range(num_episodes):
        s = env.reset()
        done = False
        total_r = 0.0
        while not done:
            a = int(np.argmax(Q[s]))
            s, r, done, _ = env.step(a)
            total_r += r
        total_returns.append(total_r)
    avg_return = np.mean(total_returns)
    print(f"Average return over {num_episodes} greedy episodes: {avg_return:.3f}")
    return avg_return


# ======================
# Pygame setup
# ======================
def animate_with_pygame_one_episode(env, Q, fps=5):
    """
    Run EXACTLY ONE episode. Count steps.
    Flash walls when the agent tries to move into them.
    """

    pygame.init()

    CELL_SIZE = 80
    n_rows, n_cols = env.n_rows, env.n_cols
    width, height = n_cols * CELL_SIZE, n_rows * CELL_SIZE

    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Slippery Maze - RL Agent")

    clock = pygame.time.Clock()

    # Colors
    COLOR_BG    = (30, 30, 30)
    COLOR_EMPTY = (220, 220, 220)
    COLOR_WALL  = (60, 60, 60)
    COLOR_WALL_HIT = (255, 215, 0)    # gold highlight
    COLOR_START = (135, 206, 250)
    COLOR_GOAL  = (144, 238, 144)
    COLOR_HOLE  = (255, 99, 71)
    COLOR_AGENT = (65, 105, 225)
    COLOR_GRID  = (180, 180, 180)
    COLOR_TEXT  = (255, 255, 255)

    font = pygame.font.SysFont(None, 40)

    # NEW: track last wall hit position
    last_wall_hit = None

    def draw_grid(state):
        screen.fill(COLOR_BG)
        agent_r, agent_c = env._state_to_pos(state)

        for r in range(n_rows):
            for c in range(n_cols):
                ch = env.grid[r][c]
                rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)

                # Pick color
                if ch == "#":
                    if last_wall_hit == (r, c):
                        color = COLOR_WALL_HIT      # highlight!
                    else:
                        color = COLOR_WALL
                elif ch == "S":
                    color = COLOR_START
                elif ch == "G":
                    color = COLOR_GOAL
                elif ch == "H":
                    color = COLOR_HOLE
                else:
                    color = COLOR_EMPTY

                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, COLOR_GRID, rect, 1)

                # agent
                if (r, c) == (agent_r, agent_c):
                    inset = CELL_SIZE // 6
                    agent_rect = pygame.Rect(
                        c * CELL_SIZE + inset,
                        r * CELL_SIZE + inset,
                        CELL_SIZE - 2 * inset,
                        CELL_SIZE - 2 * inset
                    )
                    pygame.draw.rect(screen, COLOR_AGENT, agent_rect)

        pygame.display.flip()

    def show_message(text, frames=fps*2):
        overlay = pygame.Surface((width, height))
        overlay.set_alpha(180)
        overlay.fill((0, 0, 0))
        screen.blit(overlay, (0, 0))

        surf = font.render(text, True, COLOR_TEXT)
        rect = surf.get_rect(center=(width // 2, height // 2))
        screen.blit(surf, rect)
        pygame.display.flip()

        for _ in range(frames):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            clock.tick(fps)

    # Run ONE attempt
    state = env.reset()
    done = False
    steps = 0

    last_wall_hit = None

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        draw_grid(state)

        action = int(np.argmax(Q[state]))

        # BEFORE stepping, detect if next cell is a wall
        r, c = env._state_to_pos(state)
        if action == 0:   nr, nc = r - 1, c
        elif action == 1: nr, nc = r, c + 1
        elif action == 2: nr, nc = r + 1, c
        else:             nr, nc = r, c - 1

        # Check if the attempted move hits a wall
        if not (0 <= nr < n_rows and 0 <= nc < n_cols):
            last_wall_hit = None
        elif env.grid[nr][nc] == "#":
            last_wall_hit = (nr, nc)
        else:
            last_wall_hit = None

        # Step the environment normally
        next_state, reward, done, _ = env.step(action)
        steps += 1
        state = next_state

        clock.tick(fps)

    # Show final result
    cell = env._cell_type(state)
    if cell == "G":
        msg = f"Goal! Reached in {steps} steps."
    elif cell == "H":
        msg = f"Fell in hole after {steps} steps."
    else:
        msg = f"Timeout after {steps} steps."

    draw_grid(state)
    show_message(msg)

    pygame.quit()




# ======================
# Main script
# ======================

def main():
    env = SlipperyMazeEnv(slip_prob=0.4, max_steps=200)

    # ===========================
    # Train Q-learning
    # ===========================
    print("Training Q-learning...")
    Q_qlearn, R_qlearn, S_qlearn, Q_snapshots = train_q_learning(
        env,
        num_episodes=2000,
        snapshot_points=[100, 300, 500]
    )

    # ===========================
    # Train SARSA
    # ===========================
    print("\nTraining SARSA...")
    Q_sarsa, R_sarsa, S_sarsa = train_sarsa(
        env,
        num_episodes=2000
    )

    # ===========================
    # Plot reward curves
    # ===========================
    plt.figure()
    window = 50

    # Q-learning curve
    if len(R_qlearn) >= window:
        avg_qlearn = np.convolve(R_qlearn, np.ones(window)/window, mode='valid')
        plt.plot(avg_qlearn, label="Q-learning")
    else:
        plt.plot(R_qlearn, label="Q-learning")

    # SARSA curve
    if len(R_sarsa) >= window:
        avg_sarsa = np.convolve(R_sarsa, np.ones(window)/window, mode='valid')
        plt.plot(avg_sarsa, label="SARSA")
    else:
        plt.plot(R_sarsa, label="SARSA")

    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Average Return Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ===========================
    # Plot success rates
    # ===========================
    plt.figure()
    plt.plot(S_qlearn, label="Q-learning")
    plt.plot(S_sarsa, label="SARSA")
    plt.xlabel("Hundreds of episodes")
    plt.ylabel("Success rate")
    plt.title("Success Rate Comparison")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.show()

    # ===========================
    # Visualize final policies
    # ===========================
    print("\nFinal Q-learning Policy:")
    env.render_policy(Q_qlearn)
    print("\nFinal SARSA Policy:")
    env.render_policy(Q_sarsa)

    # ===========================
    # Evaluate both
    # ===========================
    print("\nEvaluating final greedy Q-learning:")
    evaluate_policy(env, Q_qlearn, num_episodes=30)

    print("\nEvaluating final greedy SARSA:")
    evaluate_policy(env, Q_sarsa, num_episodes=30)

    # ===========================
    # Optional: animate both agents
    # ===========================
    print("\nAnimating Q-learning agent:")
    animate_with_pygame_one_episode(env, Q_qlearn)

    print("\nAnimating SARSA agent:")
    animate_with_pygame_one_episode(env, Q_sarsa)


if __name__ == "__main__":
    main()
