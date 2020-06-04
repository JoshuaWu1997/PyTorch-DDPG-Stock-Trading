"""
@File   :DDPG.py
@Author :JohsuaWu1997
@Date   :2020/1/30
"""
import numpy as np
import torch

from actor_critic import Actor, Critic
from ou_noise import OUNoise

cuda = torch.device('cuda')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

GAMMA = 0.9999999993340943687843739933894


def min_max_scale(data):
    data_min = torch.min(data, 0).values.view(1, -1)
    data_max = torch.max(data, 0).values.view(1, -1)
    data_max[data_max - data_min == 0] = 0
    return (data - data_min) / (data_max - data_min)


class DDPG:
    """docstring for DDPG"""

    def __init__(self, env, time_steps, hidden_dim):
        self.name = 'DDPG'  # name for uploading results
        self.scale = env.asset
        self.unit = env.unit
        self.seed = env.rd_seed

        self.time_dim = time_steps
        self.state_dim = env.observation_space.shape[1]
        self.action_dim = env.action_space.shape[0]
        print(self.state_dim,self.action_dim,self.time_dim)
        self.batch_size = 64
        self.memory_size = self.time_dim + self.batch_size * 10
        self.start_size = self.time_dim + self.batch_size * 2

        # Initialise actor & critic networks
        self.actor_network = Actor(self.time_dim, self.state_dim, self.action_dim, hidden_dim)
        self.critic_network = Critic(self.time_dim, self.state_dim, self.action_dim, hidden_dim)

        # Initialize replay buffer
        self.replay_state = torch.zeros((self.start_size - 1, 3, self.state_dim), device=cuda)
        self.replay_next_state = torch.zeros((self.start_size - 1, 3, self.state_dim), device=cuda)
        self.replay_action = torch.zeros((self.start_size - 1, 1, self.state_dim), device=cuda)
        self.replay_reward = torch.zeros((self.start_size - 1,), device=cuda)

        # Initialize a random process the Ornstein-Uhlenbeck process for action exploration
        self.exploration_noise = OUNoise(self.action_dim, sigma=0.05 / self.action_dim)
        self.initial()

    def initial(self):
        self.steps = 0
        self.action = np.zeros((self.action_dim,))
        self.replay_state = torch.zeros((self.start_size - 1, 3, self.state_dim), device=cuda)
        self.replay_next_state = torch.zeros((self.start_size - 1, 3, self.state_dim), device=cuda)
        self.replay_action = torch.zeros((self.start_size - 1, self.state_dim), device=cuda)
        self.replay_reward = torch.zeros((self.start_size - 1,), device=cuda)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def train_on_batch(self):
        # Sample a random minibatch of N transitions from replay buffer
        sample = torch.randint(self.time_dim, self.replay_reward.shape[0], [self.batch_size], device=cuda)
        index = torch.stack([sample - i for i in range(self.time_dim, 0, -1)]).t().reshape(-1)

        state_data = min_max_scale(self.replay_state[:, 0, :])
        amount_data = min_max_scale(self.replay_state[:, 2, :])
        next_state_data = min_max_scale(self.replay_next_state[:, 0, :])
        next_amount_data = min_max_scale(self.replay_next_state[:, 2, :])

        state_batch = torch.index_select(state_data, 0, index).view(self.batch_size, -1)
        amount_data = torch.index_select(amount_data, 0, sample).view(self.batch_size, -1)
        state_batch = torch.cat([state_batch, amount_data], dim=1)
        next_state_batch = torch.index_select(next_state_data, 0, index).view(self.batch_size, -1)
        next_amount_data = torch.index_select(next_amount_data, 0, sample).view(self.batch_size, -1)
        next_state_batch = torch.cat([next_state_batch, next_amount_data], dim=1)
        action_batch = torch.index_select(self.replay_action / self.unit, 0, sample)
        reward_batch = torch.index_select(self.replay_reward, 0, sample)

        # Calculate y_batch
        q_batch = self.critic_network.target_q(
            self.actor_network.target_action(next_state_batch), next_state_batch
        )
        y_batch = torch.add(reward_batch, q_batch, alpha=GAMMA).view(-1, 1)

        # train critic by minimizing the loss L
        self.critic_network.train(y_batch, action_batch, state_batch)

        # train actor by target loss
        self.actor_network.train(
            self.critic_network.critic_loss(
                self.actor_network.actor_action(state_batch), state_batch
            )
        )

        # Update target networks by soft update
        self.actor_network.update_target()
        self.critic_network.update_target()

    def perceive(self, state, action, reward, next_state, done):
        state_tensor = torch.tensor([state.tolist()], device=cuda)
        next_state_tensor = torch.tensor([next_state.tolist()], device=cuda)
        action_tensor = torch.tensor([action.tolist()], device=cuda)
        reward_tensor = torch.tensor([reward.tolist()], device=cuda)
        if self.steps < self.start_size - 1:
            self.replay_state[self.steps] = state_tensor
            self.replay_next_state[self.steps] = next_state_tensor
            self.replay_action[self.steps] = action_tensor
            self.replay_reward[self.steps] = reward
        else:
            if self.steps >= self.memory_size:
                self.replay_state = self.replay_state[1:]
                self.replay_next_state = self.replay_next_state[1:]
                self.replay_action = self.replay_action[1:]
                self.replay_reward = self.replay_reward[1:]
            self.replay_state = torch.cat((self.replay_state, state_tensor), dim=0)
            self.replay_next_state = torch.cat((self.replay_next_state, next_state_tensor), dim=0)
            self.replay_action = torch.cat((self.replay_action, action_tensor), dim=0)
            self.replay_reward = torch.cat((self.replay_reward, reward_tensor), dim=0)
        self.steps += 1

    def act(self, next_state, portfolio):
        if self.steps > self.start_size:
            next_state_data = min_max_scale(self.replay_next_state[:, 0, :])[-self.time_dim:].view(1, -1)
            next_amount_data = min_max_scale(self.replay_next_state[:, 2, :])[-1].view(1, -1)
            next_state_data = torch.cat([next_state_data, next_amount_data], dim=1)
            self.train_on_batch()
            allocation = self.actor_network.target_action(next_state_data).cpu().data.numpy().ravel()
            allocation[allocation < 0] = 0
            allocation /= sum(allocation)
            allocation = np.floor(
                portfolio * allocation / next_state[1, :] / self.unit
            ) * self.unit
            self.action = allocation
        return np.array(self.action)
