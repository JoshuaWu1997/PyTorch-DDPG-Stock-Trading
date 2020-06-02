"""
@File   :DDPG_agent.py
@Author :JohsuaWu1997
@Date   :01/05/2020
"""
import numpy as np
import pandas as pd
import torch

from DDPG import DDPG
from market import MarketEnv

cuda = torch.device('cuda')

raw_amount = pd.read_csv('./sh000016/i_amount.csv', header=0, index_col=0).values
raw_buy = pd.read_csv('./sh000016/o_buy.csv', header=0, index_col=0).values
raw_sell = pd.read_csv('./sh000016/o_sell.csv', header=0, index_col=0).values

START = 10441
END = 13899


def scale(data):
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    data_max[data_max - data_min == 0] = 1
    data = (data - data_min) / (data_max - data_min)
    return data


def train(Train_Env, Epoch):
    agent = DDPG(train_env, lb, node)
    for t in range(Epoch):
        print('epoch:', t)
        state, done = Train_Env.reset(), False
        agent.initial()
        while not done:
            action = agent.act(state, Train_Env.portfolio)
            next_state, reward, done, _ = Train_Env.step(action)
            agent.perceive(state, action, reward, next_state, done)
            state = next_state
            if Train_Env.n_step % 300 == 299:
                print(Train_Env.n_step, ':',
                      int(Train_Env.rewards[-1]), '\t',
                      int(sum(Train_Env.cost)), '\t',
                      int(Train_Env.available_cash[-1]), '\t',
                      agent.critic_network.loss.data, '\t',
                      agent.actor_network.loss.data
                      )
        total_reward = Train_Env.rewards[-1]
        total_cost = sum(Train_Env.cost)
        print('DDPG: Evaluation Average Reward:', total_reward)
        print('DDPG: Average Cost: ', total_cost)
    return agent


if __name__ == '__main__':
    lb, node, epoch = 36, 2048, 1
    buy_train = raw_buy[:START]
    sell_train = raw_sell[:START]
    amount_train = raw_amount[:START]

    train_env = MarketEnv([buy_train, sell_train, amount_train], 0)
    agent = train(train_env, epoch)
    torch.save(agent.actor_network.target.state_dict(), 'DDPG_model.pth')
