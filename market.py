import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gym import spaces

device = torch.device('cuda')

plt.ion()


class MarketEnv(gym.Env):
    def __init__(self, data, seed, asset=1000000.00, unit=100):
        self.asset = asset
        self.unit = unit
        self.rate = 5e-4
        self.short_rate = 1e-3
        self.rd_seed = seed
        self.sh000016 = data[0][:, 0].ravel()
        self.data = torch.tensor([data[i][:, 1:].tolist() for i in range(3)], device=device).permute(1, 0, 2)

        self.stock_number = data[0].shape[1] - 1
        self.sample_size = data[0].shape[0]
        self.action_space = spaces.Box(low=0, high=1, shape=(self.stock_number,))
        self.observation_space = spaces.Box(low=0, high=1, shape=(3, self.stock_number,))

    def reset(self):
        self.n_step = 0
        self.state = self.data[self.n_step, :, :]
        self.position = torch.zeros(self.stock_number, device=device)
        self.cash = torch.tensor(self.asset, device=device)
        self.portfolio = torch.tensor(self.asset, device=device)
        self.rewards = torch.tensor([self.asset] * self.sample_size, device=device)
        self.cost = torch.zeros(self.sample_size, device=device)
        self.success = []
        self.available_cash = torch.tensor([self.asset] * self.sample_size, device=device)
        self.book = []
        return self.state

    def step(self, position: torch.Tensor):
        self.n_step += 1
        self.state = self.data[self.n_step, :, :]
        amount = position - self.position
        price = self.state[1, :].view(-1)  # price to buy
        price[amount < 0] = self.state[0, :][amount < 0]  # price to sell

        transaction_buy = torch.sum((amount * price)[amount > 0] * self.rate)
        transaction_sell = -torch.sum((amount * price)[amount < 0] * (self.short_rate + self.rate))

        cost_buy = torch.sum((amount * price)[amount > 0])
        cost_sell = torch.sum((amount * price)[amount < 0])
        if self.cash < transaction_buy + cost_buy:
            self.success.append(False)
            self.cost[self.n_step] = transaction_sell
            self.position[amount < 0] = position[amount < 0]
            self.cash -= cost_sell + transaction_sell
        else:
            self.success.append(True)
            self.cost[self.n_step] = transaction_sell + transaction_buy
            self.position = position
            self.cash -= cost_sell + transaction_sell + cost_buy + transaction_buy

        portfolio = self.cash + torch.sum(self.state[0, :] * self.position)
        reward = portfolio - self.portfolio

        self.portfolio = portfolio
        self.rewards[self.n_step] = portfolio
        self.available_cash[self.n_step] = self.cash
        self.book.append(amount.cpu().numpy().ravel().tolist())
        if self.n_step == self.sample_size - 1:
            done = True
        else:
            done = False
        return self.state, reward, done, {}

    def plot(self, path=None, batch_size=1024):
        sh000016 = self.sh000016[1:] / self.sh000016[0] * self.asset
        plt.figure(figsize=(76.80, 43.20))
        plt.plot(sh000016)
        plt.plot(self.rewards.cpu().numpy().ravel())
        if path is not None:
            plt.savefig(path + '.png')
        plt.close()

    def render(self, mode='human', path=None):
        if path is not None:
            result = np.array([
                self.rewards.cpu().numpy().ravel(),
                self.cost.cpu().numpy().ravel(),
                self.available_cash.cpu().numpy().ravel(),
                self.success]).T
            pd.DataFrame(result, columns=['portfolio', 'transaction', 'cash', 'success']).to_csv(path + '-result.csv')
            pd.DataFrame(self.book).to_csv(path + '-book.csv')

    def close(self):
        pass
