# PyTorch-DDPG-Stock-Trading
An implementation of DDPG using PyTorch for algorithmic trading on Chinese SH50 stock market, from [Continuous Control with Deep Reinforcement Learning](https://arxiv.org/pdf/1509.02971.pdf).


## Environment
The reinforcement learning environment is to simulate Chinese SH50 stock market HF-trading at an average of 5s per tick. The environment is based on `gym` and optimised using PyTorch and GPU. Need only to change the target device to `cuda` or `cpu`.

The environment has several parameters to be set, for example: the initial cash is `asset`, minimum volume to be bought or sold is `unit`, the overall transaction rate is `rate` and the additional charge on short position is `short_rate` (which genuinely exists in Chinese stock market).

## Model
The Actor-Critic model is defined in `actor_critic.py` with act and target networks for them both. Complying to the original DDPG algorithm, the target networks are updated using `soft-copy`.

The train-on-data process is same as the original DDPG algorithm using SARSAs from memory buffer.
```
# Calculate y_batch
next_action_batch = self.actor_network.target_action(next_state_batch)
q_batch = self.critic_network.target_q(next_action_batch, next_state_batch)
y_batch = torch.add(reward_batch, q_batch, alpha=GAMMA).view(-1, 1)

# train actor-critic by target loss
self.actor_network.train(
    self.critic_network.train(
        y_batch, action_batch, state_batch
    )
)

# Update target networks by soft update
self.actor_network.update_target()
self.critic_network.update_target()
```

The policy gradience is fetched from the very first layer between actor & critic and directed to the actor's backward propagation.
```
# The policy mean gradience from critic
return torch.mean(self.critic_weights[0].grad[:, :self.action_dim], dim=0)
```
```
# Using policy gradience training the actor 
self.actor_weights[-1].backward(-loss_grad)
```

## Agent
`DDPG.py` is the wrapped up agent to collect memory buffer and train-on-data. Only `train_on_batch` and `perceive` are relevant to the algorithm. The random sampling is realised using a more sufficient way on cuda:
```
sample = torch.randint(self.time_dim, self.replay_reward.shape[0], [self.batch_size], device=cuda)

index = torch.stack([sample - i for i in range(self.time_dim, 0, -1)]).t().reshape(-1)
```
```
state_batch = torch.index_select(state_data, 0, index).view(self.batch_size, -1)
next_amount_data = torch.index_select(next_amount_data, 0, sample).view(self.batch_size, -1)
action_batch = torch.index_select(self.replay_action / self.unit, 0, sample)
reward_batch = torch.index_select(self.replay_reward, 0, sample)
```
## OUNoise
The OU-noise is implemented by [Flood Sung](https://github.com/rllab/rllab/blob/master/rllab/exploration_strategies/ou_strategy.py).

## Playground
`DDPG-agent.py` is the playground to interact. This repo provides the data of Chinese SH50 stock market from 17/04/2020 to 13/04/2020 for totally more than 13000 ticks.