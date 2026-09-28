#!/usr/bin/env python3
"""
ppo_reward.py
=============
Fonction de récompense pour PPO — échelle [-250, +100].

LOGIQUE :
  - Robot SUR la piste, bien centré  → reward positif, cumulable jusqu'à +100 sur l'épisode
  - Robot dévié (d élevé)            → reward négatif proportionnel
  - Robot hors piste (off_track)     → pénalité sévère immédiate

RESET :
  - Automatique si reward cumulé ≤ -250  (mauvais épisode)
  - Automatique après 2000 steps si toujours sur piste  (bon épisode)
"""

import math
from .lane_detector import LaneState


class PPORewardManager:
    """
    Calcule le reward step-par-step et décide si l'épisode doit se terminer.

    Échelle : reward step individuel dans [-5, +0.5]
              reward cumulé surveillé entre -250 et +100
    """

    # ── Hyperparamètres du reward step ──
    K_D          = 5.0    # pénalise fortement l'écart latéral
    K_THETA      = 2.5    # pénalise le désalignement angulaire

    # Bonus / pénalités
    ON_TRACK_BONUS      =  0.30   # robot sur piste + 2 bordures visibles
    ONE_BORDER_BONUS    =  0.05   # robot sur piste, 1 seule bordure
    OFF_TRACK_PENALTY   = -5.00   # robot complètement hors piste (step)
    NO_DETECT_PENALTY   = -0.80   # sur piste noire mais aucune bordure détectée

    # Bornes de reset sur reward cumulé
    CUMULATIVE_MIN      = -250.0  # reset automatique si on atteint ce seuil
    CUMULATIVE_MAX      =  100.0  # plafond (informatif)

    # Steps maximum si l'épisode se passe bien
    MAX_STEPS_ON_TRACK  = 2000

    def __init__(self):
        self.cumulative_reward: float = 0.0
        self.no_detect_count:   int   = 0
        self.steps_on_track:    int   = 0   # steps consécutifs sur piste

    def reset(self):
        self.cumulative_reward = 0.0
        self.no_detect_count   = 0
        self.steps_on_track    = 0

    def compute(self, lane_state: LaneState, omega: float = 0.0) -> float:
        """
        Calcule et retourne le reward du step courant.
        Met également à jour le reward cumulé.
        """

        # ── CAS 1 : hors piste (zone non-noire) ──
        if not lane_state.on_track:
            self.no_detect_count  += 1
            self.steps_on_track    = 0
            r = self.OFF_TRACK_PENALTY
            self.cumulative_reward += r
            return r

        # ── CAS 2 : sur piste noire mais détection échouée ──
        if not lane_state.valid:
            self.no_detect_count += 1
            r = self.NO_DETECT_PENALTY
            self.cumulative_reward += r
            return r

        # Détection valide
        self.no_detect_count = 0
        self.steps_on_track += 1

        d     = lane_state.d
        theta = lane_state.theta

        # Score positionnement : gaussienne sur d et theta
        r_pos   = math.exp(-self.K_D     * d**2)          # 0..1
        r_angle = math.exp(-self.K_THETA * theta**2)       # 0..1

        # ── CAS 3 : une seule bordure visible ──
        if not lane_state.both_borders:
            r = r_pos * 0.10 + self.ONE_BORDER_BONUS       # ~0.05..0.15
            self.cumulative_reward += r
            return r

        # ── CAS 4 : deux bordures, robot centré ──
        r = r_pos * r_angle * self.ON_TRACK_BONUS           # ~0.0..0.30
        self.cumulative_reward += r
        return r

    def is_done(self, lane_state: LaneState) -> bool:
        """
        Retourne True si l'épisode doit se terminer.

        Conditions de reset :
          1. Reward cumulé ≤ -250   → mauvais épisode, reset immédiat
          2. 2000 steps consécutifs sur piste  → bon épisode terminé normalement
          3. Hors piste pendant trop longtemps  → reset préventif rapide
        """
        # Condition 1 : pénalité cumulée trop élevée
        if self.cumulative_reward <= self.CUMULATIVE_MIN:
            return True

        # Condition 2 : épisode réussi, durée maximale atteinte
        if self.steps_on_track >= self.MAX_STEPS_ON_TRACK:
            return True

        # Condition 3 : hors piste depuis trop longtemps
        if not lane_state.on_track and self.no_detect_count >= 15:
            return True

        return False
