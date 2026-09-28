#!/usr/bin/env python3
"""
reward.py
=========
Fonction de récompense pour le suivi de piste circulaire.

HIÉRARCHIE DES CAS (du pire au meilleur) :
  1. Hors piste (zone non-noire détectée)  → reward très négatif
  2. Aucune détection de bordure           → reward négatif
  3. Une seule bordure visible             → reward légèrement négatif
  4. Deux bordures, robot centré (d≈0)     → reward positif (maximum ~1.1)
"""

import math
from .lane_detector import LaneState

# ── Hyperparamètres ──
K_D           = 4.0    # pénalise l'écart latéral
K_THETA       = 2.0    # pénalise le désalignement angulaire
SPEED_BONUS   = 0.1    # bonus constant pour avancer

# Pénalités — toutes NÉGATIVES, ordonnées du pire au moins grave
OFF_TRACK_PENALTY  = -10.0   # robot hors de la zone noire (sol visible, herbe, etc.)
NO_DETECT_PEN      =  -2.0   # sur piste noire mais aucune bordure visible
ONE_BORDER_PEN     =  -0.5   # une seule bordure visible (bord de piste)

DONE_D_THRESHOLD            = 1.0
MAX_STEPS_WITHOUT_DETECTION = 200


class RewardManager:

    def __init__(self):
        self.no_detect_count: int = 0

    def reset(self):
        self.no_detect_count = 0

    def compute(self, lane_state: LaneState, omega: float = 0.0) -> float:

        # ── CAS 1 : robot hors de la piste noire ──
        # on_track=False signifie que la zone sous le robot n'est PAS noire
        if not lane_state.on_track:
            self.no_detect_count += 1
            return OFF_TRACK_PENALTY      # -10.0  ← très négatif

        # ── CAS 2 : sur la piste mais détection échouée ──
        if not lane_state.valid:
            self.no_detect_count += 1
            return NO_DETECT_PEN          # -2.0

        # Détection valide → reset compteur
        self.no_detect_count = 0

        d     = lane_state.d
        theta = lane_state.theta

        # ── CAS 3 : une seule bordure visible (robot près du bord) ──
        if not lane_state.both_borders:
            r_lane = math.exp(-K_D * d**2)
            return float(r_lane * 0.1 + ONE_BORDER_PEN)  # entre -0.5 et -0.4

        # ── CAS 4 : deux bordures visibles, robot bien sur la piste ──
        r_lane  = math.exp(-K_D     * d**2)
        r_angle = math.exp(-K_THETA * theta**2)
        return float(r_lane * r_angle + SPEED_BONUS)     # entre +0.1 et +1.1

    def is_done(self, lane_state: LaneState) -> bool:
        # Fin si hors piste trop longtemps
        if not lane_state.on_track:
            return self.no_detect_count >= 10   # 10 steps hors piste → reset rapide
        if not lane_state.valid:
            return self.no_detect_count >= MAX_STEPS_WITHOUT_DETECTION
        return abs(lane_state.d) > DONE_D_THRESHOLD


# ── Fonctions libres conservées pour compatibilité ──
def compute_reward(lane_state: LaneState, omega: float = 0.0) -> float:
    raise RuntimeError("Utiliser RewardManager.compute().")

def is_done(lane_state: LaneState) -> bool:
    raise RuntimeError("Utiliser RewardManager.is_done().")