import numpy as np
import random
import pygame
from collections import deque   
import heapq     


# PARAMETERS

GRID_SIZE = 20
CELL_SIZE = 40
PANEL_WIDTH = 300

ANIMATION_FRAMES = 4
WALL_THICKNESS = 3

# Rewards (environment)
TREASURE_REWARD = 20
TRAP_PENALTY = -25
COSTLY_PATH_PENALTY = -8

BASE_STEP_PENALTY = -2
NEW_CELL_BONUS = +2
PROGRESS_BONUS = +2
LOOP_PENALTY = -3
RECENT_MEMORY = 30

# Q-learning
EPISODES = 2000
MAX_STEPS = 500
LEARNING_RATE = 0.1
DISCOUNT = 0.9
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.05

# Episodes that will be animated
VISIBLE_EPISODES = {1} | set(range(200, EPISODES + 1, 200))
VISIBLE_MAX_STEPS = 80

# Probabilities for traps & costly tiles
TRAP_PROB = 0.07
COSTLY_PROB = 0.12

# Directions
DIRS = {'N': (-1, 0), 'S': (1, 0), 'W': (0, -1), 'E': (0, 1)}
OPP = {'N': 'S', 'S': 'N', 'W': 'E', 'E': 'W'}


# Globals

env = None
agent = None
episode_rewards = []
view_mode = "visual"
show_optimal = False

astar_path = None
optimal_steps = None
optimal_cost = None

q_steps = 0
q_reached_goal = False
q_total_cost = 0.0  # cost along greedy Q run 



# MAZE GENERATION

def generate_perfect_maze(size):
    passages = [[{d: False for d in DIRS} for _ in range(size)] for _ in range(size)]
    visited = [[False] * size for _ in range(size)]
    stack = [(0, 0)]
    visited[0][0] = True

    while stack:
        r, c = stack[-1]
        neigh = []
        for d, (dr, dc) in DIRS.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size and not visited[nr][nc]:
                neigh.append((d, nr, nc))
        if neigh:
            d, nr, nc = random.choice(neigh)
            passages[r][c][d] = True
            passages[nr][nc][OPP[d]] = True
            visited[nr][nc] = True
            stack.append((nr, nc))
        else:
            stack.pop()
    return passages


def braid_maze(passages, open_prob=0.2):
    size = len(passages)
    for r in range(size):
        for c in range(size):
            for d, (dr, dc) in DIRS.items():
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size:
                    if not passages[r][c][d] and random.random() < open_prob:
                        passages[r][c][d] = True
                        passages[nr][nc][OPP[d]] = True
    return passages


def bfs_all_distances(passages, goal):
    size = len(passages)
    q = deque([goal])
    dist = {goal: 0}
    while q:
        r, c = q.popleft()
        for d, (dr, dc) in DIRS.items():
            if passages[r][c][d]:
                nr, nc = r + dr, c + dc
                if (nr, nc) not in dist:
                    dist[(nr, nc)] = dist[(r, c)] + 1
                    q.append((nr, nc))
    return dist


def shortest_path_from_start(passages, start, goal):
    q = deque([start])
    parent = {start: None}
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            path = [goal]
            while parent[path[-1]] is not None:
                path.append(parent[path[-1]])
            return list(reversed(path))
        for d, (dr, dc) in DIRS.items():
            if passages[r][c][d]:
                nr, nc = r + dr, c + dc
                if (nr, nc) not in parent:
                    parent[(nr, nc)] = (r, c)
                    q.append((nr, nc))
    return None



# COST-AWARE OPTIMAL PLANNER 

def astar_cost_aware(passages, grid, start, goal):
    """
    Full-information shortest-path planner:
    - respects walls (passages)
    - knows traps/costly tiles (grid)
    - minimizes cumulative COST, where
        reward_step ≈ BASE_STEP_PENALTY + grid[next_cell]
      so use:
        step_cost = -(BASE_STEP_PENALTY + grid[next_cell])
    NOTE: Intentionally ignore shaping bonuses (NEW_CELL_BONUS, PROGRESS_BONUS, LOOP_PENALTY)
          because those are training helpers, not true environment "terrain" costs.
    """

    def h(a, b):
        # admissible lower bound: Manhattan distance * minimal per-step cost
        # minimal step cost happens on a normal tile: -(BASE_STEP_PENALTY + 0)
        min_step_cost = -(BASE_STEP_PENALTY + 0)
        return (abs(a[0] - b[0]) + abs(a[1] - b[1])) * min_step_cost

    pq = [(h(start, goal), 0.0, start)]
    parent = {start: None}
    g = {start: 0.0}

    while pq:
        _, gc, u = heapq.heappop(pq)

        if u == goal:
            path = [u]
            while parent[u] is not None:
                u = parent[u]
                path.append(u)
            path.reverse()
            return path, gc

        r, c = u
        for d, (dr, dc) in DIRS.items():
            if passages[r][c][d]:
                v = (r + dr, c + dc)

                step_cost = -(BASE_STEP_PENALTY + grid[v])  # COST = -terrain_reward
                ng = gc + float(step_cost)

                if ng < g.get(v, 1e18):
                    g[v] = ng
                    parent[v] = u
                    heapq.heappush(pq, (ng + h(v, goal), ng, v))

    return None, None


class Environment:
    def __init__(self):
        while True:
            passages = braid_maze(generate_perfect_maze(GRID_SIZE))
            start = (0, 0)

            # Ensure maze is sufficiently connected
            dist_start = bfs_all_distances(passages, start)
            if len(dist_start) < GRID_SIZE * GRID_SIZE // 3:
                continue

            # Choose farthest cell as goal
            goal = max(dist_start.items(), key=lambda x: x[1])[0]

            # Create terrain grid
            grid = np.zeros((GRID_SIZE, GRID_SIZE))
            grid[goal] = TREASURE_REWARD

            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    if (r, c) == start or (r, c) == goal:
                        continue
                    p = random.random()
                    if p < TRAP_PROB:
                        grid[r, c] = TRAP_PENALTY
                    elif p < TRAP_PROB + COSTLY_PROB:
                        grid[r, c] = COSTLY_PATH_PENALTY

            # Ensure reachability (walls only)
            dist_goal = bfs_all_distances(passages, goal)
            if start not in dist_goal:
                continue

            # Accept environment
            self.passages = passages
            self.grid = grid
            self.goal = goal
            self.dist_map = dist_goal
            break

        self.agent_pos = [0, 0]
        self.visited_ever = set()
        self.recent_positions = deque(maxlen=RECENT_MEMORY)

    def reset(self):
        self.agent_pos = [0, 0]
        self.visited_ever = {(0, 0)}
        self.recent_positions.clear()
        self.recent_positions.append((0, 0))
        return (0, 0)

    def dist(self, cell):
        return self.dist_map.get(cell, 9999)

    def step(self, a):
        r, c = self.agent_pos
        d = ['N', 'S', 'W', 'E'][a]
        dr, dc = DIRS[d]
        old = (r, c)
        nr, nc = r, c

        if self.passages[r][c][d]:
            nr, nc = r + dr, c + dc
            self.agent_pos = [nr, nc]

        new = (nr, nc)

        reward = BASE_STEP_PENALTY + self.grid[nr, nc]
        done = False

        if new == self.goal:
            reward += TREASURE_REWARD
            done = True

        if new not in self.visited_ever:
            reward += NEW_CELL_BONUS
            self.visited_ever.add(new)

        if new in self.recent_positions:
            reward += LOOP_PENALTY

        self.recent_positions.append(new)

        if self.dist(new) < self.dist(old):
            reward += PROGRESS_BONUS

        return new, reward, done



class QAgent:
    def __init__(self):
        self.Q = np.zeros((GRID_SIZE, GRID_SIZE, 4))

    def act(self, s, eps):
        if random.random() < eps:
            return random.randint(0, 3)
        return self.greedy(s)

    def greedy(self, s):
        q = self.Q[s]
        m = np.max(q)
        best = np.flatnonzero(q == m)
        return int(np.random.choice(best))

    def update(self, s, a, r, ns):
        self.Q[s][a] = (
            (1 - LEARNING_RATE) * self.Q[s][a]
            + LEARNING_RATE * (r + DISCOUNT * np.max(self.Q[ns]))
        )


pygame.init()

MAZE_WIDTH = GRID_SIZE * CELL_SIZE
SCREEN_WIDTH = MAZE_WIDTH + PANEL_WIDTH
SCREEN_HEIGHT = GRID_SIZE * CELL_SIZE

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Treasure Maze Q-Learning with Cost-aware baseline")
clock = pygame.time.Clock()

font = pygame.font.SysFont(None, 18)
font_q = pygame.font.SysFont(None, 14)

# Images
grass = pygame.transform.scale(pygame.image.load("grass.png"), (CELL_SIZE, CELL_SIZE))
rock = pygame.transform.scale(pygame.image.load("rock.png"), (CELL_SIZE - 8, CELL_SIZE - 8))
trap_img = pygame.transform.scale(pygame.image.load("spikes.png"), (CELL_SIZE, CELL_SIZE))
treas = pygame.transform.scale(pygame.image.load("treasure.png"), (CELL_SIZE - 8, CELL_SIZE - 8))
try:
    agent_img = pygame.transform.scale(pygame.image.load("agent.png"), (CELL_SIZE - 8, CELL_SIZE - 8))
    use_agent = True
except:
    agent_img = None
    use_agent = False
    print("agent.png not found, using blue circle.")



def draw_agent(px, py):
    if use_agent:
        screen.blit(agent_img, (px + 4, py + 4))
    else:
        pygame.draw.circle(
            screen,
            (0, 0, 255),
            (px + CELL_SIZE // 2, py + CELL_SIZE // 2),
            CELL_SIZE // 3,
        )


def draw_side_panel():
    """Right-side info panel: controls and stats."""
    panel_x = MAZE_WIDTH
    pygame.draw.rect(screen, (0, 0, 0), (panel_x, 0, PANEL_WIDTH, SCREEN_HEIGHT))

    y = 10
    def line(text, color=(255, 255, 255)):
        nonlocal y
        txt = font.render(text, True, color)
        screen.blit(txt, (panel_x + 10, y))
        y += 22

    line("Controls:")
    line("R: New maze & train")
    line("T: Retrain same maze")
    line("D: Debug view")
    line("V: Visual view")
    line("O: Toggle optimal (cost-aware)")
    y += 10

    line("Stats:")
    if optimal_steps is not None:
        line(f"Optimal steps: {optimal_steps}")
    else:
        line("Optimal steps: N/A", (200, 80, 80))

    if optimal_cost is not None:
        line(f"Optimal cost : {optimal_cost:.1f}")
    else:
        line("Optimal cost : N/A", (200, 80, 80))

    if q_reached_goal and q_steps > 0:
        line(f"Q steps      : {q_steps}")
        line(f"Q cost       : {q_total_cost:.1f}")
        eff = (optimal_cost / q_total_cost * 100.0) if (optimal_cost is not None and q_total_cost > 0) else 0.0
        line(f"Efficiency   : {eff:.1f}%")
    else:
        line("Q: did not reach", (200, 80, 80))


def draw_astar_overlay(path):
    if not path or len(path) < 2:
        return
    pts = []
    for (r, c) in path:
        x = c * CELL_SIZE + CELL_SIZE // 2
        y = r * CELL_SIZE + CELL_SIZE // 2
        pts.append((x, y))
    pygame.draw.lines(screen, (255, 255, 0), False, pts, 3)  # yellow


def draw_q_path(path):
    if not path or len(path) < 2:
        return
    pts = []
    for (r, c) in path:
        x = c * CELL_SIZE + CELL_SIZE // 2
        y = r * CELL_SIZE + CELL_SIZE // 2
        pts.append((x, y))
    pygame.draw.lines(screen, (0, 150, 255), False, pts, 3)  # blue


def draw_visual(path):
    # Maze on left
    screen.fill((0, 0, 0))
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            x, y = c * CELL_SIZE, r * CELL_SIZE
            screen.blit(grass, (x, y))
            val = env.grid[r, c]
            if val == TRAP_PENALTY:
                screen.blit(trap_img, (x, y))
            elif val == COSTLY_PATH_PENALTY:
                screen.blit(rock, (x + 4, y + 4))
            elif (r, c) == env.goal:
                screen.blit(treas, (x + 4, y + 4))

    # Walls
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            x, y = c * CELL_SIZE, r * CELL_SIZE
            p = env.passages[r][c]
            if not p['N']:
                pygame.draw.line(screen, (0, 0, 0), (x, y), (x + CELL_SIZE, y), WALL_THICKNESS)
            if not p['W']:
                pygame.draw.line(screen, (0, 0, 0), (x, y), (x, y + CELL_SIZE), WALL_THICKNESS)
            if not p['S']:
                pygame.draw.line(screen, (0, 0, 0), (x, y + CELL_SIZE), (x + CELL_SIZE, y + CELL_SIZE), WALL_THICKNESS)
            if not p['E']:
                pygame.draw.line(screen, (0, 0, 0), (x + CELL_SIZE, y), (x + CELL_SIZE, y + CELL_SIZE), WALL_THICKNESS)

    # Paths
    if show_optimal and astar_path:
        draw_astar_overlay(astar_path)
    draw_q_path(path)

    draw_side_panel()


def draw_debug(current_pos=None):
    screen.fill((0, 0, 0))
    maxq = np.max(agent.Q, axis=2)
    qmin = np.min(maxq)
    qmax = np.max(maxq)
    if qmax - qmin < 1e-6:
        qmax = qmin + 1e-6

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            x, y = c * CELL_SIZE, r * CELL_SIZE
            val = env.grid[r, c]
            if val == TRAP_PENALTY:
                base = (80, 0, 0)
            elif val == COSTLY_PATH_PENALTY:
                base = (80, 40, 0)
            elif (r, c) == env.goal:
                base = (100, 80, 0)
            else:
                base = (10, 50, 10)
            pygame.draw.rect(screen, base, (x, y, CELL_SIZE, CELL_SIZE))

            v = maxq[r, c]
            norm = (v - qmin) / (qmax - qmin)
            red = int((1 - norm) * 255)
            green = int(norm * 255)
            overlay = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
            overlay.fill((red, green, 0, 70))
            screen.blit(overlay, (x, y))

            qvals = agent.Q[r, c]
            u = font_q.render(f"{qvals[0]:+.1f}", True, (255, 255, 255))
            d = font_q.render(f"{qvals[1]:+.1f}", True, (255, 255, 255))
            l = font_q.render(f"{qvals[2]:+.1f}", True, (255, 255, 255))
            rr = font_q.render(f"{qvals[3]:+.1f}", True, (255, 255, 255))

            screen.blit(u, (x + CELL_SIZE / 2 - 12, y + 1))
            screen.blit(d, (x + CELL_SIZE / 2 - 12, y + CELL_SIZE - 17))
            screen.blit(l, (x + 2, y + CELL_SIZE / 2 - 7))
            screen.blit(rr, (x + CELL_SIZE - 24, y + CELL_SIZE / 2 - 7))

            pygame.draw.rect(screen, (30, 30, 30), (x, y, CELL_SIZE, CELL_SIZE), 1)

    if current_pos:
        r, c = current_pos
        pygame.draw.rect(screen, (0, 0, 255),
                         (c * CELL_SIZE + 3, r * CELL_SIZE + 3, CELL_SIZE - 6, CELL_SIZE - 6), 2)

    draw_side_panel()


def train_fast():
    global agent, episode_rewards, view_mode
    agent = QAgent()
    episode_rewards = []
    eps = 1.0

    for ep in range(1, EPISODES + 1):
        s = env.reset()
        total = 0
        visible = ep in VISIBLE_EPISODES
        max_steps = VISIBLE_MAX_STEPS if visible else MAX_STEPS

        for step in range(max_steps):
            # Handle events so D/V work even during animation
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_d:
                        view_mode = "debug"
                    elif event.key == pygame.K_v:
                        view_mode = "visual"

            a = agent.act(s, eps)
            ns, r, d = env.step(a)
            agent.update(s, a, r, ns)
            s = ns
            total += r

            if visible:
                if view_mode == "debug":
                    draw_debug(current_pos=s)
                else:
                    draw_visual(path=[s])
                pygame.display.flip()
                clock.tick(20)

            if d:
                break

        eps = max(MIN_EPSILON, eps * EPSILON_DECAY)
        episode_rewards.append(total)



def reset_test():
    global vis_state, vis_path, done, q_steps, q_reached_goal, q_total_cost
    vis_state = env.reset()
    vis_path = [vis_state]
    done = False
    q_steps = 0
    q_total_cost = 0.0
    q_reached_goal = False


def step_test():
    global vis_state, vis_path, done, q_steps, q_reached_goal, q_total_cost

    if done:
        return

    a = agent.greedy(vis_state)
    ns, r, d = env.step(a)

    # Track COST in a fair way: terrain only (base + tile), ignore shaping
    step_cost = -(BASE_STEP_PENALTY + env.grid[ns])
    q_total_cost += float(step_cost)

    q_steps += 1
    vis_state = ns
    vis_path.append(ns)

    if d:
        q_reached_goal = True
        done = True
    elif q_steps >= MAX_STEPS:
        done = True

    if view_mode == "visual":
        draw_visual(vis_path)
        draw_agent(ns[1] * CELL_SIZE, ns[0] * CELL_SIZE)
        pygame.display.flip()
        clock.tick(45)
    else:
        draw_debug(current_pos=ns)
        pygame.display.flip()
        clock.tick(25)



# NEW MAZE / RETRAIN

def compute_optimal_baseline():
    """Compute cost-aware optimal path and its cost/steps."""
    global astar_path, optimal_steps, optimal_cost
    path, cost = astar_cost_aware(env.passages, env.grid, (0, 0), env.goal)
    astar_path = path
    optimal_steps = (len(path) - 1) if path else None
    optimal_cost = cost


def new_maze():
    global env, show_optimal
    show_optimal = False
    env = Environment()
    train_fast()
    compute_optimal_baseline()
    reset_test()


def retrain_same_maze():
    global show_optimal
    show_optimal = False
    train_fast()
    compute_optimal_baseline()
    reset_test()



# MAIN

env = Environment()
train_fast()
compute_optimal_baseline()
reset_test()

running = True
while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_r:
                new_maze()
            elif e.key == pygame.K_t:
                retrain_same_maze()
            elif e.key == pygame.K_d:
                view_mode = "debug"
            elif e.key == pygame.K_v:
                view_mode = "visual"
            elif e.key == pygame.K_o:
                show_optimal = not show_optimal

    if not done:
        step_test()
    else:
        if view_mode == "visual":
            draw_visual(vis_path)
            draw_agent(env.agent_pos[1] * CELL_SIZE, env.agent_pos[0] * CELL_SIZE)
        else:
            draw_debug(current_pos=tuple(env.agent_pos))
        pygame.display.flip()
        clock.tick(30)

pygame.quit()
