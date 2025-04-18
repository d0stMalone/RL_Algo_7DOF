import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=256, final_activation=None):
        super().__init__()
        layers = [
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)
        ]
        if final_activation:
            layers.append(final_activation)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU()
        )
        self.mean = nn.Linear(256, act_dim)
        self.log_std = nn.Linear(256, act_dim)
        self.act_limit = act_limit
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, obs):
        x = self.net(obs)
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(self.log_std_min, self.log_std_max)
        std = log_std.exp()
        return mean, std

    def sample(self, obs):
        mean, std = self.forward(obs)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.act_limit

        # Log-prob correction for tanh squashing
        log_prob = normal.log_prob(x_t).sum(dim=1)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6).sum(dim=1)
        return action, log_prob, mean

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

class SACAgent:
    def __init__(self, obs_dim, act_dim, act_limit, gamma=0.99, tau=0.005, alpha=0.2, auto_entropy=True, target_entropy=None):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.auto_entropy = auto_entropy

        self.actor = GaussianPolicy(obs_dim, act_dim, act_limit)
        self.critic1 = MLP(obs_dim + act_dim, 1)
        self.critic2 = MLP(obs_dim + act_dim, 1)
        self.critic1_target = MLP(obs_dim + act_dim, 1)
        self.critic2_target = MLP(obs_dim + act_dim, 1)

        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=3e-4)
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=3e-4)

        if self.auto_entropy:
            if target_entropy is None:
                self.target_entropy = -act_dim
            else:
                self.target_entropy = target_entropy
            self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True)
            self.alpha_opt = optim.Adam([self.log_alpha], lr=3e-4)

        self.replay = ReplayBuffer()
        self.batch_size = 256
        self.act_limit = act_limit

    def select_action(self, obs, eval_mode=False):
        obs = torch.FloatTensor(obs).unsqueeze(0)
        if eval_mode:
            with torch.no_grad():
                mean, _ = self.actor.forward(obs)
                return torch.tanh(mean)[0].cpu().numpy() * self.act_limit
        else:
            with torch.no_grad():
                action, _, _ = self.actor.sample(obs)
                return action[0].cpu().numpy()

    def store(self, s, a, r, s2, d):
        self.replay.add((s, a, r, s2, float(d)))

    def update(self):
        if len(self.replay.storage) < self.batch_size:
            return

        s, a, r, s2, d = self.replay.sample(self.batch_size)

        with torch.no_grad():
            next_a, next_log_prob, _ = self.actor.sample(s2)
            q1_next = self.critic1_target(torch.cat([s2, next_a], dim=1))
            q2_next = self.critic2_target(torch.cat([s2, next_a], dim=1))
            min_q = torch.min(q1_next, q2_next) - self.alpha * next_log_prob.unsqueeze(1)
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

        # Actor update
        sampled_a, log_prob, _ = self.actor.sample(s)
        q1_val = self.critic1(torch.cat([s, sampled_a], dim=1))
        q2_val = self.critic2(torch.cat([s, sampled_a], dim=1))
        min_q_val = torch.min(q1_val, q2_val)
        actor_loss = (self.alpha * log_prob - min_q_val.squeeze()).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Entropy tuning
        if self.auto_entropy:
            entropy_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            entropy_loss.backward()
            self.alpha_opt.step()
            self.alpha = self.log_alpha.exp().item()

        # Soft target update
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
