import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, max_size=1_000_000):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, transition):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = transition
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.storage, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            torch.FloatTensor(s),
            torch.FloatTensor(a),
            torch.FloatTensor(r).unsqueeze(1),
            torch.FloatTensor(s2),
            torch.FloatTensor(d).unsqueeze(1),
        )

class DDPGAgent:
    def __init__(self, obs_dim, act_dim, act_limit, gamma=0.99, tau=0.005, actor_lr=1e-4, critic_lr=1e-3):
        self.gamma = gamma
        self.tau = tau
        self.act_limit = act_limit

        self.actor = MLP(obs_dim, act_dim)
        self.actor_target = MLP(obs_dim, act_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = MLP(obs_dim + act_dim, 1)
        self.critic_target = MLP(obs_dim + act_dim, 1)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.replay = ReplayBuffer()
        self.batch_size = 256

    def select_action(self, obs, noise_scale=0.1):
        obs = torch.FloatTensor(obs).unsqueeze(0)
        action = self.actor(obs).detach().cpu().numpy()[0]
        action += noise_scale * np.random.randn(len(action))
        return np.clip(action, -self.act_limit, self.act_limit)

    def store(self, s, a, r, s2, d):
        self.replay.add((s, a, r, s2, float(d)))

    def update(self):
        if len(self.replay.storage) < self.batch_size:
            return

        s, a, r, s2, d = self.replay.sample(self.batch_size)

        with torch.no_grad():
            a2 = self.actor_target(s2)
            q2 = self.critic_target(torch.cat([s2, a2], dim=1))
            target_q = r + (1 - d) * self.gamma * q2

        # Critic update
        current_q = self.critic(torch.cat([s, a], dim=1))
        critic_loss = nn.MSELoss()(current_q, target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Actor update
        actor_loss = -self.critic(torch.cat([s, self.actor(s)], dim=1)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Soft target update
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
