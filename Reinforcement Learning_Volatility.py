# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 15:20:55 2024

@author: WONCHAN
"""

import gym
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from stable_baselines3 import DQN


ticker = '^KS200'
data = yf.download(ticker, start='2000-01-01', end='2024-12-31')

data['Returns'] = data['Adj Close'].pct_change()
data.dropna(inplace=True)

class MarketEnv(gym.Env):
    def __init__(self, returns):
        super(MarketEnv, self).__init__()
        self.returns = returns
        self.current_step = 0
        self.action_space = gym.spaces.Discrete(3)  # 매수, 매도, 유지
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        return np.array([self.returns[self.current_step]])

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.returns) - 1
        reward = -abs(self.returns[self.current_step])  # 변동성 예측의 정확도를 보상으로 설정
        state = np.array([self.returns[self.current_step]]) if not done else None
        return state, reward, done, {}

env = MarketEnv(data['Returns'].values)

model = DQN('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

states = []
rewards = []
state = env.reset()
done = False

while not done:
    action, _states = model.predict(state)
    state, reward, done, info = env.step(action)
    if state is not None:
        states.append(state[0])
    rewards.append(reward)

plt.figure(figsize=(14, 7))
plt.plot(rewards, label='Rewards')
plt.title('Rewards Over Time')
plt.xlabel('Time Step')
plt.ylabel('Reward')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 7))
plt.plot(data.index[1:], data['Returns'].values[1:], label='Actual Returns')
plt.plot(data.index[1:len(states) + 1], states, label='Predicted Returns')
plt.title('Actual vs Predicted Returns')
plt.xlabel('Date')
plt.ylabel('Returns')
plt.legend()
plt.grid(True)
plt.show()


