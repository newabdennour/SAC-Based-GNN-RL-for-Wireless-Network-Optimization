import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sac_graph5g.config import DEVICE

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        s, a, r, ns, d = map(np.asarray, zip(*batch))
        return (
            torch.tensor(s, dtype=torch.float32, device=DEVICE), 
            torch.tensor(a, dtype=torch.float32, device=DEVICE), 
            torch.tensor(r, dtype=torch.float32, device=DEVICE).unsqueeze(1), 
            torch.tensor(ns, dtype=torch.float32, device=DEVICE), 
            torch.tensor(d, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        )
        
    def __len__(self):
        return len(self.buffer)

class SACActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(), 
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.mu = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)
        
    def sample(self, state):
        h = self.net(state)
        mu = self.mu(h)
        std = self.log_std(h).clamp(-5, 2).exp()
        normal = torch.distributions.Normal(mu, std)
        z = normal.rsample()
        action = torch.tanh(z)
        logp = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        return action, logp.sum(dim=1, keepdim=True)

class SACCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(), 
            nn.Linear(hidden, hidden), nn.ReLU(), 
            nn.Linear(hidden, 1)
        )
        
    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=1))

def soft_update(target, source, tau=0.025):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.mul_(1 - tau).add_(tau * sp.data)

def sac_update(actor, q1, q2, tq1, tq2, actor_opt, q1_opt, q2_opt, replay, batch_size=32, gamma=0.92, alpha=0.12):
    if len(replay) < 8:
        return
    state, action, reward, next_state, done = replay.sample(batch_size)
    with torch.no_grad():
        next_action, next_logp = actor.sample(next_state)
        target_q = torch.min(tq1(next_state, next_action), tq2(next_state, next_action)) - alpha * next_logp
        target = reward + gamma * (1 - done) * target_q
        
    q1_loss = F.mse_loss(q1(state, action), target)
    q2_loss = F.mse_loss(q2(state, action), target)
    
    q1_opt.zero_grad(set_to_none=True)
    q1_loss.backward()
    q1_opt.step()
    
    q2_opt.zero_grad(set_to_none=True)
    q2_loss.backward()
    q2_opt.step()
    
    new_action, logp = actor.sample(state)
    actor_loss = (alpha * logp - torch.min(q1(state, new_action), q2(state, new_action))).mean()
    
    actor_opt.zero_grad(set_to_none=True)
    actor_loss.backward()
    actor_opt.step()
    
    soft_update(tq1, q1)
    soft_update(tq2, q2)
