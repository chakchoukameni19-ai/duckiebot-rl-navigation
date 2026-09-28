#!/usr/bin/env python3
"""
reward_odom.py
==============
Récompense basée uniquement sur /odom.

La piste est un anneau circulaire centré en (0, 0) :
  • Rayon cible  : TRACK_RADIUS = 1.8 m  (position de spawn)
  • Largeur piste : TRACK_HALF_WIDTH = 0.20 m  (marge tolérable)

État fourni : [x, y, yaw, vx, omega]  → dimension 5

Récompense :
  r = exp(-k * dist_err²) + bonus_vitesse   si dans la piste
  r = OUT_PENALTY                           si hors piste
"""

import math

# ── Géométrie de la piste ──
TRACK_RADIUS     = 1.8    # rayon de l'axe central de la piste (m)
TRACK_HALF_WIDTH = 0.22   # demi-largeur tolérable (m)  → zone valide : [1.58, 2.02]

# ── Hyperparamètres récompense ──
K_DIST       = 8.0    # pénalise l'écart au rayon cible
K_ALIGN      = 2.0    # pénalise le mauvais alignement tangentiel
SPEED_BONUS  = 0.15   # encourage à avancer
OUT_PENALTY  = -5.0   # sortie de piste

# ── Seuils done ──
DONE_DIST_THRESHOLD  = TRACK_HALF_WIDTH * 1.5   # ~0.33 m d'écart → terminé
MAX_STEPS_PER_EP     = 600                        # sécurité épisode infini


class OdomState:
    """État extrait de /odom."""
    __slots__ = ("x", "y", "yaw", "vx", "omega",
                 "dist_err", "align_err")

    def __init__(self, x=0.0, y=0.0, yaw=0.0, vx=0.0, omega=0.0):
        self.x     = x
        self.y     = y
        self.yaw   = yaw
        self.vx    = vx
        self.omega = omega

        # Calculés une seule fois
        r              = math.hypot(x, y)
        self.dist_err  = r - TRACK_RADIUS          # écart signé au rayon cible (m)
        # Angle tangentiel idéal = atan2(x, -y) + π  (sens trigonométrique)
        # align_err = différence entre yaw du robot et tangente locale
        tangent_angle  = math.atan2(x, -y)         # direction tangente au cercle
        self.align_err = _angle_diff(yaw, tangent_angle)

    def to_array(self):
        """Retourne le vecteur d'état normalisé pour le DQN."""
        import numpy as np
        return np.array([
            self.x     / 3.0,                          # normalisé sur ~[-1,1]
            self.y     / 3.0,
            self.yaw   / math.pi,                      # [-1, 1]
            float(np.clip(self.vx    / 0.5,  -1, 1)),  # vmax ≈ 0.5 m/s
            float(np.clip(self.omega / 2.0,  -1, 1)),  # ωmax ≈ 2 rad/s
        ], dtype=np.float32)


def _angle_diff(a: float, b: float) -> float:
    """Différence angulaire dans [-π, π]."""
    d = (a - b + math.pi) % (2 * math.pi) - math.pi
    return d


class RewardManager:
    """Gère la récompense et la condition done (sans état global)."""

    def __init__(self):
        self.steps: int = 0

    def reset(self):
        self.steps = 0

    def compute(self, state: OdomState) -> float:
        dist = abs(state.dist_err)

        # Hors piste → forte pénalité
        if dist > DONE_DIST_THRESHOLD:
            return OUT_PENALTY

        # Récompense gaussienne sur l'écart radial + alignement
        r_dist  = math.exp(-K_DIST  * state.dist_err**2)
        r_align = math.exp(-K_ALIGN * state.align_err**2)
        return float(r_dist * r_align + SPEED_BONUS)

    def is_done(self, state: OdomState) -> bool:
        self.steps += 1
        if abs(state.dist_err) > DONE_DIST_THRESHOLD:
            return True
        if self.steps >= MAX_STEPS_PER_EP:
            return True
        return False
