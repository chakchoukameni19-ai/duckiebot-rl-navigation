#!/usr/bin/env python3
"""
lane_detector.py
================
Détecte les lignes blanches de la piste circulaire et calcule :
  • d     : distance latérale au centre de la piste  [-1, 1]
              positif = robot à droite du centre
  • theta : angle entre l'axe du robot et la tangente à la piste  [-1, 1]
  • on_track : True si le robot est bien SUR la zone noire (piste)

Stratégie :
  1. Convertir l'image en HSV
  2. Masquer les pixels blancs (bordures) et noirs (route)
  3. Vérifier que le bas de l'image (sous le robot) est majoritairement NOIR
     → si non : robot hors piste → valid=False, on_track=False
  4. Chercher les deux bordures (gauche/droite) sur une bande horizontale
  5. Estimer d et theta à partir des positions des bordures

Compatible ROS2 / sensor_msgs/Image via cv_bridge.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class LaneState:
    d:            float = 0.0   # distance latérale normalisée [-1, 1]
    theta:        float = 0.0   # angle normalisé              [-1, 1]
    valid:        bool  = False  # détection réussie ET robot sur piste noire
    both_borders: bool  = False  # True seulement si les DEUX bordures sont visibles
    on_track:     bool  = False  # True si la zone sous le robot est noire (piste)


class LaneDetector:
    """
    Détection légère basée sur la couleur et la position des bordures blanches.
    Inclut une vérification que le robot est bien sur la zone noire.
    """

    # ── Seuillage HSV ──
    # Blanc (bordures de piste)
    WHITE_LOW  = np.array([0,   0, 180], dtype=np.uint8)
    WHITE_HIGH = np.array([180, 50, 255], dtype=np.uint8)

    # Noir (surface de la piste)
    # HSV : teinte quelconque, saturation faible, VALUE très basse
    BLACK_LOW  = np.array([0,   0,   0], dtype=np.uint8)
    BLACK_HIGH = np.array([180, 255, 60], dtype=np.uint8)

    # Seuil : au moins X% des pixels de la zone centrale doivent être noirs
    ON_TRACK_BLACK_RATIO = 0.35   # 35 % de noir → robot sur la piste

    def __init__(self, track_width_px: int = 200):
        self.track_width_px = track_width_px

    # ──────────────────────────────────────────
    def detect(self, image_bgr: np.ndarray) -> LaneState:
        """
        Prend une image BGR (640×480) et renvoie un LaneState.
        """
        h, w = image_bgr.shape[:2]
        cx = w // 2

        # 1. Convertir HSV
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

        # ── VÉRIFICATION CRITIQUE : le robot est-il sur la piste noire ? ──
        # On analyse le tiers inférieur central de l'image (zone sous le robot)
        check_r1 = int(h * 0.60)
        check_r2 = int(h * 0.95)
        check_c1 = int(w * 0.25)
        check_c2 = int(w * 0.75)
        roi_hsv = hsv[check_r1:check_r2, check_c1:check_c2]

        black_mask_roi = cv2.inRange(roi_hsv, self.BLACK_LOW, self.BLACK_HIGH)
        black_ratio = np.sum(black_mask_roi > 0) / black_mask_roi.size

        on_track = black_ratio >= self.ON_TRACK_BLACK_RATIO

        # Si pas sur piste noire → retourner immédiatement invalid
        if not on_track:
            return LaneState(valid=False, on_track=False)

        # 2. Masque pixels blancs (bordures)
        mask = cv2.inRange(hsv, self.WHITE_LOW, self.WHITE_HIGH)

        # 3. Morphologie pour supprimer le bruit
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 4. Bande de scan
        row_start = int(h * 0.60)
        row_end   = int(h * 0.85)
        scan_band = mask[row_start:row_end, :]

        # 5. Projection horizontale
        col_presence = np.sum(scan_band, axis=0)
        threshold = scan_band.shape[0] * 0.15

        left_col  = self._find_border(col_presence, cx, direction="left",  thr=threshold)
        right_col = self._find_border(col_presence, cx, direction="right", thr=threshold)

        # 6. Si aucune bordure détectée malgré qu'on soit sur la piste
        if left_col is None and right_col is None:
            return LaneState(valid=False, on_track=True)

        both_borders = (left_col is not None and right_col is not None)

        if both_borders:
            center_detected = (left_col + right_col) / 2.0
        elif left_col is not None:
            center_detected = left_col + self.track_width_px / 2
        else:
            center_detected = right_col - self.track_width_px / 2  # type: ignore

        # d : écart au centre image (normalisé)
        d = (center_detected - cx) / (w / 2)
        d = float(np.clip(d, -1.0, 1.0))

        # theta
        theta = self._estimate_angle(mask, h, w)

        return LaneState(d=d, theta=theta, valid=True,
                         both_borders=both_borders, on_track=True)

    # ──────────────────────────────────────────
    @staticmethod
    def _find_border(
        col_presence: np.ndarray,
        cx: int,
        direction: str,
        thr: float,
    ) -> Optional[int]:
        if direction == "left":
            indices = np.where(col_presence[:cx] > thr)[0]
            return int(indices[-1]) if len(indices) > 0 else None
        else:
            indices = np.where(col_presence[cx:] > thr)[0]
            return int(indices[0]) + cx if len(indices) > 0 else None

    # ──────────────────────────────────────────
    @staticmethod
    def _estimate_angle(mask: np.ndarray, h: int, w: int) -> float:
        def row_center(r_start, r_end):
            band = mask[r_start:r_end, :]
            cols = np.where(np.sum(band, axis=0) > band.shape[0] * 0.1)[0]
            return float(np.mean(cols)) if len(cols) > 0 else None

        top_center = row_center(int(h * 0.40), int(h * 0.55))
        bot_center = row_center(int(h * 0.65), int(h * 0.80))

        if top_center is None or bot_center is None:
            return 0.0

        diff = (bot_center - top_center) / (w / 2)
        return float(np.clip(diff, -1.0, 1.0))


# ──────────────────────────────────────────
# Visualisation (debug uniquement)
# ──────────────────────────────────────────
def draw_debug(image_bgr: np.ndarray, state: LaneState) -> np.ndarray:
    vis = image_bgr.copy()
    h, w = vis.shape[:2]
    cx = w // 2

    if state.valid:
        detected_cx = int(cx + state.d * (w / 2))
        cv2.line(vis, (detected_cx, h - 80), (detected_cx, h - 10), (0, 255, 0), 3)
        cv2.line(vis, (cx, h - 80), (cx, h - 10), (255, 0, 0), 2)

    color = (0, 200, 0) if state.on_track else (0, 0, 255)
    cv2.putText(vis,
                f"d={state.d:.3f}  theta={state.theta:.3f}  "
                f"valid={state.valid}  on_track={state.on_track}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return vis