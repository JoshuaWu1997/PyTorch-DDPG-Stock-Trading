"""
@File   :market.py
@Author :JohsuaWu1997
@Date   :2020/1/30
"""
import gym
from gym import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.ion()


class MarketEnv(gym.Env):
    def __init__(self, data, seed, asset=1000000, unit=100):
        self.asset = asset
        self.unit = unit
        self.rate = 5e-4
        self.short_rate = 1e-3
        self.rd_seed = seed
        self.sh000016 = data[0][:, 0].ravel()
        self.data = np.asarray([data[i][:, 1:].tolist() for i in range(3)])
        self.data = np.swapaxes(self.data, 0, 1)

        self.stock_number = self.data.shape[2]
        self.sample_size = self.data.shape[0]
        self.action_space = spaces.Box(low=0, high=1, shape=(self.stock_number,))
        self.observation_space = spaces.Box(low=0, high=1, shape=(3, self.stock_number,))

    def reset(self):
        self.n_step = 0
        self.state = self.data[self.n_step, :, :]
        self.position = np.zeros(shape=(self.stock_number,))
        self.cash = self.asset
        self.portfolio = self.asset
        self.rewards = []
        self.cost = []
        self.success = []
        self.available_cash = []
        self.book = []
        return self.state

    def step(self, action: np.ndarray):
        self.n_step += 1
        self.state = self.data[self.n_step, :, :]
        amount = action - self.position
        price = self.state[1, :].ravel()  # price to buy
        price[amount < 0] = self.state[0, :].ravel()[amount < 0]  # price to sell

        transaction_buy = np.sum((amount * price)[amount > 0] * self.rate)
        transaction_sell = -np.sum((amount * price)[amount < 0] * (self.short_rate + self.rate)) 
        '''
        transaction_buy = 0
        transaction_sell = 0
        '''
        cost_buy = np.sum((amount * price)[amount > 0])
        cost_sell = np.sum((amount * price)[amount < 0])
        if self.cash < transaction_buy + cost_buy:
            self.success.append(False)
            self.cost.append(transaction_sell)
            self.position[amount < 0] = action[amount < 0]
            self.cash -= cost_sell + transaction_sell
        else:
            self.success.append(True)
            self.cost.append(transaction_sell + transaction_buy)
            self.position = action
            self.cash -= cost_sell + transaction_sell + cost_buy + transaction_buy

        portfolio = self.cash + np.sum(self.state[0, :] * self.position)
        reward = portfolio - self.portfolio
        '''
        if np.isnan(portfolio):
            print(action)
        '''
        self.portfolio = portfolio
        self.rewards.append(portfolio)
        self.available_cash.append(self.cash)
        self.book.append(amount)
        if self.n_step == self.sample_size - 1:
            done = True
        else:
            done = False
        return self.state, reward, done, {}

    def plot(self, path=None, batch_size=1024):
        sh000016 = self.sh000016[1:] / self.sh000016[0] * self.asset
        plt.figure(figsize=(76.80, 43.20))
        plt.plot(sh000016)
        plt.plot(self.rewards)
        if path is not None:
            plt.savefig(path + '.png')
        plt.close()

    def render(self, mode='human', path=None):
        if path is not None:
            result = np.array([self.rewards, self.cost, self.available_cash, self.success]).T
            pd.DataFrame(result, columns=['portfolio', 'transaction', 'cash', 'success']).to_csv(path + '-result.csv')
            pd.DataFrame(self.book).to_csv(path + '-book.csv')

    def close(self):
        pass
