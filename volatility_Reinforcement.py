# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 20:26:30 2024

@author: WONCHAN
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
from collections import deque
import random

class OptionPricingEnv:
    def __init__(self):
        self.state_size = 2  # 주가와 남은 시간
        self.action_size = 3  # 매수, 매도, 아무 것도 안 함
        self.reset()

    def reset(self):
        self.stock_price = np.random.uniform(50, 150)  # 임의의 초기 주가
        self.time_to_maturity = np.random.uniform(0, 1)  # 임의의 초기 만기 시간
        self.done = False
        return np.array([self.stock_price, self.time_to_maturity])

    def step(self, action):
        if self.done:
            raise ValueError("Episode already done")
        
        # 간단한 가격 변화 및 시간 감소 모델
        self.stock_price += np.random.normal(0, 1)
        self.time_to_maturity -= 0.01

        # 보상 계산 (간단한 예: 옵션 가격의 변화로 보상 설정)
        reward = 0
        if action == 0:  # 매수
            reward = max(self.stock_price - 100, 0)  # 콜 옵션의 가치
        elif action == 1:  # 매도
            reward = max(100 - self.stock_price, 0)  # 풋 옵션의 가치

        if self.time_to_maturity <= 0:
            self.done = True

        next_state = np.array([self.stock_price, self.time_to_maturity])
        return next_state, reward, self.done

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = models.Sequential()
        model.add(layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=optimizers.Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# 환경 설정 및 학습 루프
env = OptionPricingEnv()
state_size = env.state_size
action_size = env.action_size
agent = DQNAgent(state_size, action_size)
done = False
batch_size = 32

EPISODES = 1000  # 학습 에피소드 수

for e in range(EPISODES):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    for time in range(500):
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print("episode: {}/{}, score: {}, e: {:.2}".format(e, EPISODES, time, agent.epsilon))
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

