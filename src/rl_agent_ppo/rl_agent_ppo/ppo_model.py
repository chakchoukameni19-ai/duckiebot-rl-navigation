#!/usr/bin/env python3
"""
ppo_model.py
============
Actor-Critic réseau + agent PPO pour suivi de piste circulaire.

Architecture :
  - État  : [d, theta, omega]  (dim=3)
  - Acteur : distribution continue sur (linear_vel, angular_vel)
  - Critique : valeur scalaire V(s)

PPO avec clipping, entropie, et normalisation des avantages.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


# ─────────────────────────────────────────────
# Réseau Actor-Critic partagé
# ─────────────────────────────────────────────
class ActorCritic(nn.Module):
    """
    Corps partagé → tête Actor (μ, log_σ) + tête Critique (V).
    Même backbone que DQN (3 → 128 → 128) pour cohérence.
    """

    def __init__(self, state_dim: int = 3):
        super().__init__()

        # Corps partagé
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )
        # Tête acteur : μ pour [linear_vel, angular_vel]
        self.actor_mean = nn.Linear(128, 2)

        # log_std comme paramètre appris (indépendant de l'état)
        self.log_std = nn.Parameter(torch.zeros(2))

        # Tête critique : valeur scalaire
        self.critic = nn.Linear(128, 1)

        # Init des poids
        self._init_weights()

    def _init_weights(self):
        for m in self.shared:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.zeros_(self.actor_mean.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, x: torch.Tensor):
        feat = self.shared(x)
        mean = self.actor_mean(feat)
        std  = self.log_std.exp().clamp(1e-4, 1.0)
        value = self.critic(feat).squeeze(-1)
        return mean, std, value

    def get_action_and_value(self, state: torch.Tensor, action=None):
        mean, std, value = self.forward(state)
        dist = Normal(mean, std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(-1)   # sum sur [lin, ang]
        entropy  = dist.entropy().sum(-1)

        return action, log_prob, entropy, value

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        _, _, value = self.forward(state)
        return value


# ─────────────────────────────────────────────
# Tampon de rollout (trajectoire courante)
# ─────────────────────────────────────────────
class RolloutBuffer:
    """
    Stocke une trajectoire complète avant chaque mise à jour PPO.
    """

    def __init__(self, capacity: int, state_dim: int = 3, device: str = "cpu"):
        self.capacity  = capacity
        self.device    = torch.device(device)
        self.state_dim = state_dim
        self.reset()

    def reset(self):
        self.states    = torch.zeros(self.capacity, self.state_dim)
        self.actions   = torch.zeros(self.capacity, 2)
        self.rewards   = torch.zeros(self.capacity)
        self.dones     = torch.zeros(self.capacity)
        self.log_probs = torch.zeros(self.capacity)
        self.values    = torch.zeros(self.capacity)
        self.ptr       = 0
        self.full      = False

    def push(self, state, action, reward, done, log_prob, value):
        idx = self.ptr % self.capacity
        self.states[idx]    = torch.FloatTensor(state)
        self.actions[idx]   = torch.FloatTensor(action)
        self.rewards[idx]   = float(reward)
        self.dones[idx]     = float(done)
        self.log_probs[idx] = float(log_prob)
        self.values[idx]    = float(value)
        self.ptr += 1
        if self.ptr >= self.capacity:
            self.full = True

    def is_ready(self) -> bool:
        return self.full

    def compute_returns_and_advantages(self, last_value: float, gamma: float, lam: float):
        """GAE-Lambda : calcule returns et avantages."""
        returns    = torch.zeros(self.capacity)
        advantages = torch.zeros(self.capacity)

        last_gae   = 0.0
        last_val   = last_value

        for t in reversed(range(self.capacity)):
            mask   = 1.0 - self.dones[t].item()
            delta  = self.rewards[t].item() + gamma * last_val * mask - self.values[t].item()
            last_gae = delta + gamma * lam * mask * last_gae
            advantages[t] = last_gae
            last_val = self.values[t].item()

        returns = advantages + self.values
        return returns, advantages

    def get_all(self):
        return (
            self.states.to(self.device),
            self.actions.to(self.device),
            self.log_probs.to(self.device),
            self.values.to(self.device),
        )


# ─────────────────────────────────────────────
# Agent PPO
# ─────────────────────────────────────────────
class PPOAgent:
    """
    Proximal Policy Optimization avec :
      • Clipping (ε=0.2)
      • Coefficient entropie pour l'exploration
      • Normalisation des avantages
      • Gradient clipping
    """

    # Bornes physiques des actions
    LIN_MIN, LIN_MAX = 0.05, 0.25
    ANG_MIN, ANG_MAX = -1.0,  1.0

    def __init__(
        self,
        state_dim:       int   = 3,
        lr:              float = 3e-4,
        gamma:           float = 0.99,
        lam:             float = 0.95,    # GAE lambda
        clip_eps:        float = 0.2,     # PPO clip
        value_coef:      float = 0.5,     # coefficient perte valeur
        entropy_coef:    float = 0.01,    # bonus entropie
        n_epochs:        int   = 4,       # passes sur le buffer
        batch_size:      int   = 64,
        rollout_len:     int   = 256,     # steps avant update
        device:          str   = "cpu",
    ):
        self.gamma        = gamma
        self.lam          = lam
        self.clip_eps     = clip_eps
        self.value_coef   = value_coef
        self.entropy_coef = entropy_coef
        self.n_epochs     = n_epochs
        self.batch_size   = batch_size
        self.rollout_len  = rollout_len
        self.device       = torch.device(device)

        self.net       = ActorCritic(state_dim).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)

        self.buffer = RolloutBuffer(rollout_len, state_dim, device)

        # Stats de la dernière update
        self.last_policy_loss: float = 0.0
        self.last_value_loss:  float = 0.0
        self.last_entropy:     float = 0.0
        self.last_total_loss:  float = 0.0
        self.update_count:     int   = 0

    # ── Sélection d'action ──
    @torch.no_grad()
    def select_action(self, state: np.ndarray):
        """
        Retourne (action_clampée, log_prob, value).
        action = [linear_vel, angular_vel]
        """
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action, log_prob, _, value = self.net.get_action_and_value(s)

        action_np = action.squeeze(0).cpu().numpy()

        # Clamp physique
        action_np[0] = float(np.clip(action_np[0], self.LIN_MIN, self.LIN_MAX))
        action_np[1] = float(np.clip(action_np[1], self.ANG_MIN, self.ANG_MAX))

        return action_np, float(log_prob.item()), float(value.item())

    # ── Stockage ──
    def store(self, state, action, reward, done, log_prob, value):
        self.buffer.push(state, action, reward, done, log_prob, value)

    def buffer_ready(self) -> bool:
        return self.buffer.is_ready()

    # ── Mise à jour PPO ──
    def learn(self, last_state: np.ndarray) -> dict:
        """
        Déclenché quand le buffer est plein.
        Retourne un dict avec les métriques de l'update.
        """
        # Valeur de bootstrap de la dernière observation
        with torch.no_grad():
            ls = torch.FloatTensor(last_state).unsqueeze(0).to(self.device)
            last_value = float(self.net.get_value(ls).item())

        returns, advantages = self.buffer.compute_returns_and_advantages(
            last_value, self.gamma, self.lam
        )
        returns    = returns.to(self.device)
        advantages = advantages.to(self.device)

        # Normaliser les avantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states, actions, old_log_probs, _ = self.buffer.get_all()

        total_policy_loss = 0.0
        total_value_loss  = 0.0
        total_entropy     = 0.0
        n_updates         = 0

        for _ in range(self.n_epochs):
            # Mini-batches aléatoires
            idx = torch.randperm(self.rollout_len)
            for start in range(0, self.rollout_len, self.batch_size):
                mb_idx = idx[start:start + self.batch_size]

                mb_states    = states[mb_idx]
                mb_actions   = actions[mb_idx]
                mb_old_lp    = old_log_probs[mb_idx]
                mb_adv       = advantages[mb_idx]
                mb_returns   = returns[mb_idx]

                _, new_log_probs, entropy, new_values = \
                    self.net.get_action_and_value(mb_states, mb_actions)

                # Ratio de probabilités
                ratio = (new_log_probs - mb_old_lp).exp()

                # PPO clipped loss
                surr1 = ratio * mb_adv
                surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Valeur loss
                value_loss = nn.MSELoss()(new_values, mb_returns)

                # Total
                loss = policy_loss \
                     + self.value_coef * value_loss \
                     - self.entropy_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss  += value_loss.item()
                total_entropy     += entropy.mean().item()
                n_updates         += 1

        self.buffer.reset()
        self.update_count += 1

        n = max(n_updates, 1)
        self.last_policy_loss = total_policy_loss / n
        self.last_value_loss  = total_value_loss  / n
        self.last_entropy     = total_entropy     / n
        self.last_total_loss  = (self.last_policy_loss
                                 + self.value_coef * self.last_value_loss
                                 - self.entropy_coef * self.last_entropy)

        return {
            "policy_loss": self.last_policy_loss,
            "value_loss":  self.last_value_loss,
            "entropy":     self.last_entropy,
            "total_loss":  self.last_total_loss,
            "update_count": self.update_count,
        }

    # ── Sauvegarde / chargement ──
    def save(self, path: str):
        torch.save({
            "net":          self.net.state_dict(),
            "optimizer":    self.optimizer.state_dict(),
            "update_count": self.update_count,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.update_count = ckpt.get("update_count", 0)
        self.net.eval()
