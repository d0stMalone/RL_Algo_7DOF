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
            self.storage[self.ptr] = transition
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

class TD3Agent:
    def __init__(self, obs_dim, act_dim, act_limit, gamma=0.99, tau=0.005, noise_std=0.2, noise_clip=0.5, delay=2):
        self.gamma = gamma
        self.tau = tau
        self.act_limit = act_limit
        self.noise_std = noise_std
        self.noise_clip = noise_clip
        self.policy_delay = delay
        self.total_updates = 0

        # Actor
        self.actor = MLP(obs_dim, act_dim)
        self.actor_target = MLP(obs_dim, act_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Two Critics (for double Q-learning)
        self.critic1 = MLP(obs_dim + act_dim, 1)
        self.critic2 = MLP(obs_dim + act_dim, 1)
        self.critic1_target = MLP(obs_dim + act_dim, 1)
        self.critic2_target = MLP(obs_dim + act_dim, 1)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Optimizers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=1e-3)
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=1e-3)

        self.replay = ReplayBuffer()
        self.batch_size = 256

    def select_action(self, obs, noise_scale=0.1):
        obs = torch.FloatTensor(obs).unsqueeze(0)
        action = self.actor(obs).detach().cpu().numpy()[0]
        noise = noise_scale * np.random.randn(len(action))
        return np.clip(action + noise, -self.act_limit, self.act_limit)

    def store(self, s, a, r, s2, d):
        self.replay.add((s, a, r, s2, float(d)))

    def update(self):
        if len(self.replay.storage) < self.batch_size:
            return

        s, a, r, s2, d = self.replay.sample(self.batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(a) * self.noise_std).clamp(-self.noise_clip, self.noise_clip)
            a2 = (self.actor_target(s2) + noise).clamp(-self.act_limit, self.act_limit)

            q1_target = self.critic1_target(torch.cat([s2, a2], dim=1))
            q2_target = self.critic2_target(torch.cat([s2, a2], dim=1))
            min_q = torch.min(q1_target, q2_target)
            target_q = r + (1 - d) * self.gamma * min_q

        # Critic 1 update
        current_q1 = self.critic1(torch.cat([s, a], dim=1))
        critic1_loss = nn.MSELoss()(current_q1, target_q)
        self.critic1_opt.zero_grad()
        critic1_loss.backward()
        self.critic1_opt.step()

        # Critic 2 update
        current_q2 = self.critic2(torch.cat([s, a], dim=1))
        critic2_loss = nn.MSELoss()(current_q2, target_q)
        self.critic2_opt.zero_grad()
        critic2_loss.backward()
        self.critic2_opt.step()

        # Delayed actor update
        if self.total_updates % self.policy_delay == 0:
            actor_loss = -self.critic1(torch.cat([s, self.actor(s)], dim=1)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            # Soft update targets
            for p, pt in zip(self.actor.parameters(), self.actor_target.parameters()):
                pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)
            for p, pt in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)
            for p, pt in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)

        self.total_updates += 1
