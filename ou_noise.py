# --------------------------------------
# Ornstein-Uhlenbeck Noise
# Author: Flood Sung
# Date: 2016.5.4
# Reference: https://github.com/rllab/rllab/blob/master/rllab/exploration_strategies/ou_strategy.py
# --------------------------------------

import numpy as np


class OUNoise:
    """docstring for OUNoise"""

    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.seeds = 0
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu
        self.seeds += 1
        np.random.seed(self.seeds)

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

    def std_noise(self, mu, sigma):
        self.seeds += 1
        np.random.seed(self.seeds)
        return np.random.normal(mu, sigma, len(self.state))


if __name__ == '__main__':
    ou = OUNoise(1, sigma=0.1 / 50)
    states = []
    for i in range(1000):
        states.append(ou.noise())
    import matplotlib.pyplot as plt

    # plt.plot(states)
    plt.hist(np.array(states).ravel())
    plt.show()
