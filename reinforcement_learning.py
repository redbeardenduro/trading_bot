import numpy as np
import pandas as pd
import gym
from gym import spaces
from stable_baselines3 import PPO

class TradingEnv(gym.Env):
    """A custom environment for reinforcement learning trading."""
    def __init__(self, data, initial_balance=10000):
        super(TradingEnv, self).__init__()
        self.data = data
        self.initial_balance = initial_balance
        self.current_step = 0
        self.action_space = spaces.Discrete(3)  # Buy, Hold, Sell
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(data.columns),), dtype=np.float32)

    def reset(self):
        self.balance = self.initial_balance
        self.positions = 0
        self.current_step = 0
        return self._next_observation()

    def _next_observation(self):
        return self.data.iloc[self.current_step].values

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        reward = 0
        
        if action == 0:  # Buy
            self.positions += self.balance / self.data.iloc[self.current_step]['close']
            self.balance = 0
        elif action == 2:  # Sell
            self.balance += self.positions * self.data.iloc[self.current_step]['close']
            self.positions = 0

        reward = self.balance + self.positions * self.data.iloc[self.current_step]['close'] - self.initial_balance

        return self._next_observation(), reward, done, {}

    def render(self, mode='human'):
        pass

def train_rl_model(data):
    env = TradingEnv(data)
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10000)
    return model
