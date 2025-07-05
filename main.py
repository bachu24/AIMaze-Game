import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class MazeEnv:
    def __init__(self, maze, start, goal, max_steps=100):
        self.maze = np.array(maze)
        self.start = start
        self.goal = goal
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.pos = self.start
        self.steps = 0
        return self._get_state()

    def step(self, action):
        moves = [(-1,0), (1,0), (0,-1), (0,1)]  # Up, Down, Left, Right
        dx, dy = moves[action]
        new_x, new_y = self.pos[0] + dx, self.pos[1] + dy
        self.steps += 1

        if (0 <= new_x < self.maze.shape[0]) and (0 <= new_y < self.maze.shape[1]):
            if self.maze[new_x, new_y] == 1:
                reward = -10
            else:
                self.pos = (new_x, new_y)
                reward = -1
        else:
            reward = -10

        done = False
        if self.pos == self.goal:
            reward = 100
            done = True
        if self.steps >= self.max_steps:
            done = True

        return self._get_state(), reward, done

    def _get_state(self):
        # State: flatten maze, agent pos, goal pos (one-hot)
        state = np.zeros(self.maze.shape, dtype=np.float32)
        state[self.pos] = 1.0
        state[self.goal] = 0.5
        return state.flatten()

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity

    def push(self, transition):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

def train_dqn(env, episodes=500, batch_size=64, gamma=0.99, lr=1e-3, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995):
    n_actions = 4
    state_dim = env._get_state().shape[0]
    policy_net = DQN(state_dim, n_actions)
    target_net = DQN(state_dim, n_actions)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    buffer = ReplayBuffer(10000)
    epsilon = epsilon_start

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            if random.random() < epsilon:
                action = random.randint(0, n_actions-1)
            else:
                with torch.no_grad():
                    q_vals = policy_net(torch.tensor(state).float().unsqueeze(0))
                    action = q_vals.argmax().item()
            next_state, reward, done = env.step(action)
            buffer.push((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            # Train
            if len(buffer) >= batch_size:
                batch = buffer.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                states = torch.tensor(states).float()
                actions = torch.tensor(actions).long()
                rewards = torch.tensor(rewards).float()
                next_states = torch.tensor(next_states).float()
                dones = torch.tensor(dones).float()

                q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
                next_q_values = target_net(next_states).max(1)[0]
                expected_q = rewards + gamma * next_q_values * (1 - dones)
                loss = nn.MSELoss()(q_values, expected_q.detach())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Update target network
        if ep % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        if (ep+1) % 50 == 0:
            print(f"Episode {ep+1}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")

    return policy_net

def get_dqn_path(env, policy_net):
    state = env.reset()
    path = [env.pos]
    done = False
    for _ in range(env.max_steps):
        with torch.no_grad():
            q_vals = policy_net(torch.tensor(state).float().unsqueeze(0))
            action = q_vals.argmax().item()
        state, _, done = env.step(action)
        if env.pos == path[-1]:  # stuck
            break
        path.append(env.pos)
        if done:
            break
    return path

# Example usage:
maze = [
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0]
]
start = (0, 0)
goal = (4, 4)
env = MazeEnv(maze, start, goal)
policy_net = train_dqn(env)
path = get_dqn_path(env, policy_net)
print("Learned path:", path)

# --- Visualization ---
def plot_maze_path(maze, path):
    maze = np.array(maze)
    plt.figure(figsize=(6,6))
    plt.imshow(maze, cmap='gray_r')
    path_x = [p[1] for p in path]
    path_y = [p[0] for p in path]
    plt.plot(path_x, path_y, marker='o', color='red', linewidth=2, markersize=8, label='Path')
    plt.scatter(path_x[0], path_y[0], color='green', s=100, label='Start')
    plt.scatter(path_x[-1], path_y[-1], color='blue', s=100, label='Goal')
    plt.legend()
    plt.title('Learned Path in Maze')
    plt.gca().invert_yaxis()
    plt.show()

plot_maze_path(maze, path)
