# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 14:29:32 2024

@author: WONCHAN
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

#환경 파라미터
T = 100
initial_wealth = 100
gamma = 0.99
epsilon = 0.7
epsilon_decay = 0.995
min_epsilon = 0.01
num_episodes = 500
batch_size = 64
learning_rate = 0.0001
memory_capacity = 5000

wealth_levels = 200
num_actions = 10
action_space = np.linspace(0, 1, num_actions)

#GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#Replay Memory
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

#Neural Network for DQN
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

#Environment transition and utility function
def transition(wealth, consumption, risky_weight=0.5):
    base_growth = 0.02
    risky_return = np.random.normal(0.03, 0.01)
    portfolio_return = base_growth + risky_weight * (risky_return - base_growth)
    next_wealth = wealth - consumption + portfolio_return * (wealth - consumption)
    return max(0, next_wealth)

def utility(future_wealth):
    if future_wealth < 50:
        return -1e6
    else:
        return future_wealth

#Initialize network, optimizer, and memory
input_dim = 2
output_dim = num_actions
policy_net = DQN(input_dim, output_dim).to(device)
target_net = DQN(input_dim, output_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
memory = ReplayMemory(memory_capacity)

#Epsilon-greedy action selection
def select_action(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.randrange(num_actions)
    else:
        with torch.no_grad():
            state = state.to(device)
            return policy_net(state).argmax().item()

#Train the DQN
def optimize_model():
    if len(memory) < batch_size:
        return

    transitions = memory.sample(batch_size)
    batch = list(zip(*transitions))

    state_batch = torch.tensor(batch[0], dtype=torch.float32).to(device)
    action_batch = torch.tensor(batch[1], dtype=torch.int64).unsqueeze(1).to(device)
    reward_batch = torch.tensor(batch[2], dtype=torch.float32).to(device)
    next_state_batch = torch.tensor(batch[3], dtype=torch.float32).to(device)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = target_net(next_state_batch).max(1)[0].detach()
    expected_state_action_values = reward_batch + (gamma * next_state_values)

    loss = nn.MSELoss()(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#Training loop
for episode in range(num_episodes):
    wealth = initial_wealth
    total_reward = 0

    for t in range(T):
        state = torch.tensor([wealth / 200, t / T], dtype=torch.float32).unsqueeze(0).to(device)
        action = select_action(state, epsilon)
        consumption_ratio = action_space[action]
        consumption = min(consumption_ratio * wealth, 0.02 * wealth)

        next_wealth = transition(wealth, consumption)
        reward = utility(next_wealth)

        next_state = torch.tensor([next_wealth / 200, (t + 1) / T], dtype=torch.float32).unsqueeze(0).to(device)

        memory.push((state.squeeze().cpu().tolist(), action, reward, next_state.squeeze().cpu().tolist()))

        wealth = next_wealth
        total_reward += reward

        if wealth <= 0:
            break

        if (t + 1) % 10 == 0 and len(memory) > batch_size:
            optimize_model()

    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    if (episode + 1) % 100 == 0:
        target_net.load_state_dict(policy_net.state_dict())
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward:.2f}")

print("Training complete!")

#Calculate W_100 using the trained policy
example_wealth = initial_wealth
for t in range(T):
    state = torch.tensor([example_wealth / 200, t / T], dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        action = policy_net(state).argmax().item()
    optimal_consumption_ratio = action_space[action]
    optimal_consumption = min(optimal_consumption_ratio * example_wealth, 0.02 * example_wealth)

    example_wealth = transition(example_wealth, optimal_consumption)
    if example_wealth <= 0:
        print(f"Period {t + 1}, Wealth depleted.")
        break

print(f"\nW_100 (Final Wealth): {example_wealth}")
