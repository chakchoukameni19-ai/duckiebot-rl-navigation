#!/usr/bin/env python3
"""
dqn_model_odom.py
=================
Agent DQN basé uniquement sur /odom.

État  : [x_norm, y_norm, yaw_norm, vx_norm, omega_norm]  dim=5
Actions : 5 commandes (lin, ang)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


# ─────────────────────────────────────────────
class DQNNetwork(nn.Module):
    """MLP 5 → 128 → 128 → n_actions."""

    def __init__(self, state_dim: int = 5, n_actions: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            np.array(state,      dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            np.array(s),
            np.array(a),
            np.array(r),
            np.array(ns),
            np.array(d, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────
class DQNAgent:
    """
    Double DQN + ε-greedy + Huber loss + gradient clipping.
    Adapté pour état dim=5 (odom uniquement).
    """

    # (linear_vel m/s, angular_vel rad/s)
    ACTIONS = [
        (0.20,  0.00),   # 0 : tout droit
        (0.15,  0.50),   # 1 : gauche doux
        (0.15, -0.50),   # 2 : droite doux
        (0.10,  1.00),   # 3 : gauche fort
        (0.10, -1.00),   # 4 : droite fort
    ]

    def __init__(
        self,
        state_dim:    int   = 5,
        n_actions:    int   = 5,
        lr:           float = 1e-3,
        gamma:        float = 0.95,
        eps_start:    float = 1.0,
        eps_end:      float = 0.05,
        eps_decay:    int   = 3000,
        batch_size:   int   = 64,
        target_update:int   = 200,
        buffer_size:  int   = 50_000,
        device:       str   = "cpu",
    ):
        self.n_actions     = n_actions
        self.gamma         = gamma
        self.batch_size    = batch_size
        self.target_update = target_update
        self.device        = torch.device(device)

        self.eps_start  = eps_start
        self.eps_end    = eps_end
        self.eps_decay  = eps_decay
        self.steps_done = 0

        self.online_net = DQNNetwork(state_dim, n_actions).to(self.device)
        self.target_net = DQNNetwork(state_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.buffer    = ReplayBuffer(buffer_size)

    @property
    def epsilon(self) -> float:
        return self.eps_end + (self.eps_start - self.eps_end) * \
               np.exp(-self.steps_done / self.eps_decay)

    def select_action(self, state: np.ndarray) -> int:
        self.steps_done += 1
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return int(self.online_net(s).argmax(dim=1).item())

    def get_cmd(self, action_idx: int):
        return self.ACTIONS[action_idx]

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def learn(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None

        s, a, r, ns, d = self.buffer.sample(self.batch_size)

        s  = torch.FloatTensor(s).to(self.device)
        a  = torch.LongTensor(a).unsqueeze(1).to(self.device)
        r  = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        ns = torch.FloatTensor(ns).to(self.device)
        d  = torch.FloatTensor(d).unsqueeze(1).to(self.device)

        q_values = self.online_net(s).gather(1, a)

        with torch.no_grad():
            next_q = self.target_net(ns).max(dim=1, keepdim=True)[0]
            target = r + self.gamma * next_q * (1.0 - d)

        loss = nn.SmoothL1Loss()(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return float(loss.item())

    def save(self, path: str):
        torch.save({
            "online": self.online_net.state_dict(),
            "target": self.target_net.state_dict(),
            "steps":  self.steps_done,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online"])
        self.target_net.load_state_dict(ckpt["target"])
        self.steps_done = ckpt.get("steps", 0)
        self.online_net.eval()
        self.target_net.eval()
