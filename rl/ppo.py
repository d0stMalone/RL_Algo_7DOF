import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, act_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.log_std = nn.Parameter(torch.ones(act_dim) * 0.0)

    def forward(self, x):
        value = self.critic(x)
        mean = self.actor(x)
        std = torch.exp(self.log_std)
        return mean, std, value

class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.states, self.actions, self.log_probs = [], [], []
        self.rewards, self.dones, self.values = [], [], []

    def compute_returns_and_advantages(self, last_value, gamma=0.99, gae_lambda=0.95):
        returns, advantages = [], []
        gae = 0
        values = self.values + [last_value]
        for i in reversed(range(len(self.rewards))):
            delta = self.rewards[i] + gamma * values[i+1] * (1 - self.dones[i]) - values[i]
            gae = delta + gamma * gae_lambda * (1 - self.dones[i]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])
        self.returns = returns
        self.advantages = advantages

class PPOAgent:
    def __init__(self, obs_dim, act_dim, lr=3e-4, gamma=0.99, clip_eps=0.2, update_epochs=10, batch_size=64):
        self.net = ActorCritic(obs_dim, act_dim)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.buffer = RolloutBuffer()
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.update_epochs = update_epochs
        self.batch_size = batch_size

    def select_action(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32)
        mean, std, value = self.net(obs)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action.detach().numpy(), log_prob.item(), value.item()

    def store(self, state, action, log_prob, reward, done, value):
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.log_probs.append(log_prob)
        self.buffer.rewards.append(reward)
        self.buffer.dones.append(done)
        self.buffer.values.append(value)

    def update(self, last_value):
        self.buffer.compute_returns_and_advantages(last_value, self.gamma)

        states = torch.tensor(np.array(self.buffer.states), dtype=torch.float32)
        actions = torch.tensor(np.array(self.buffer.actions), dtype=torch.float32)
        old_log_probs = torch.tensor(np.array(self.buffer.log_probs), dtype=torch.float32)
        returns = torch.tensor(np.array(self.buffer.returns), dtype=torch.float32)
        advantages = torch.tensor(np.array(self.buffer.advantages), dtype=torch.float32)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.update_epochs):
            idx = np.random.permutation(len(states))
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_idx = idx[start:end]
                s, a, olp, ret, adv = states[batch_idx], actions[batch_idx], old_log_probs[batch_idx], returns[batch_idx], advantages[batch_idx]

                mean, std, value = self.net(s)
                dist = torch.distributions.Normal(mean, std)
                log_prob = dist.log_prob(a).sum(dim=-1)

                ratio = torch.exp(log_prob - olp)
                surrogate1 = ratio * adv
                surrogate2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
                policy_loss = -torch.min(surrogate1, surrogate2).mean()

                value_loss = nn.MSELoss()(value.squeeze(), ret)
                loss = policy_loss + 0.5 * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.buffer.clear()
