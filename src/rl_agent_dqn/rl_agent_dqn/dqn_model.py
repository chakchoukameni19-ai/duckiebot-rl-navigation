#!/usr/bin/env python3
"""
dqn_model.py
============
Réseau DQN léger + Experience Replay Buffer.
- Input  : état [d, theta, omega]  (dim=3)
- Output : Q-values pour chaque action (dim=5)
Optimisé pour converger vite sur une piste circulaire simple.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


# ─────────────────────────────────────────────
# Réseau de neurones
# ─────────────────────────────────────────────
class DQNNetwork(nn.Module):
    """
    MLP 3 couches : 3 → 128 → 128 → n_actions
    Suffisant pour un état de dimension 3 et 5 actions.
    """
    def __init__(self, state_dim: int = 3, n_actions: int = 5):
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
# Replay Buffer
# ─────────────────────────────────────────────
class ReplayBuffer:
    """Circular buffer of (s, a, r, s', done) transitions."""

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
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────
# Agent DQN
# ─────────────────────────────────────────────
class DQNAgent:
    """
    Deep Q-Network avec :
      • Double DQN (online + target network)
      • ε-greedy décroissant
      • Gradient clipping pour la stabilité
    """

    # Actions : (linear_vel, angular_vel)
    ACTIONS = [
        (0.20,  0.00),   # 0 : tout droit
        (0.15,  0.40),   # 1 : virage gauche doux
        (0.15, -0.40),   # 2 : virage droite doux
        (0.10,  0.80),   # 3 : virage gauche fort
        (0.10, -0.80),   # 4 : virage droite fort
    ]

    def __init__(
        self,
        state_dim:    int   = 3,
        n_actions:    int   = 5,
        lr:           float = 1e-3,
        gamma:        float = 0.95,
        eps_start:    float = 1.0,
        eps_end:      float = 0.05,
        eps_decay:    int   = 2000,   # steps jusqu'à eps_end
        batch_size:   int   = 64,
        target_update:int   = 200,    # steps entre synchro target
        buffer_size:  int   = 50_000,
        device:       str   = "cpu",
    ):
        self.n_actions     = n_actions
        self.gamma         = gamma
        self.batch_size    = batch_size
        self.target_update = target_update
        self.device        = torch.device(device)

        # ε-greedy
        self.eps_start = eps_start
        self.eps_end   = eps_end
        self.eps_decay = eps_decay
        self.steps_done = 0

        # Réseaux
        self.online_net = DQNNetwork(state_dim, n_actions).to(self.device)
        self.target_net = DQNNetwork(state_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.buffer    = ReplayBuffer(buffer_size)

        self.loss_history: list = []

    # ── Epsilon courant ──
    @property
    def epsilon(self) -> float:
        decay = self.eps_decay
        eps = self.eps_end + (self.eps_start - self.eps_end) * \
              np.exp(-self.steps_done / decay)
        return eps

    # ── Sélection d'action ──
    def select_action(self, state: np.ndarray) -> int:
        self.steps_done += 1
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return int(self.online_net(s).argmax(dim=1).item())

    def get_cmd(self, action_idx: int):
        """Retourne (linear, angular) pour l'action choisie."""
        return self.ACTIONS[action_idx]

    # ── Stockage ──
    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    # ── Apprentissage ──
    def learn(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        s  = torch.FloatTensor(states).to(self.device)
        a  = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        r  = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        ns = torch.FloatTensor(next_states).to(self.device)
        d  = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Q(s,a) courant
        q_values = self.online_net(s).gather(1, a)

        # Target : r + γ * max Q_target(s')
        with torch.no_grad():
            next_q = self.target_net(ns).max(dim=1, keepdim=True)[0]
            target = r + self.gamma * next_q * (1.0 - d)

        loss = nn.SmoothL1Loss()(q_values, target)   # Huber loss → stable

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        # Sync target
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        val = float(loss.item())
        self.loss_history.append(val)
        return val

    # ── Sauvegarde / chargement ──
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
