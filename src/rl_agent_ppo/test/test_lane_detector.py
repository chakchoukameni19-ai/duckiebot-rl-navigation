#!/usr/bin/env python3
"""
test_lane_detector.py
=====================
Test rapide du détecteur de piste sur une image synthétique.
Lance sans ROS : python3 test_lane_detector.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import cv2

from rl_agent_ppo.lane_detector import LaneDetector, draw_debug


def make_synthetic_track(
    width=640, height=480,
    center_x=320, track_half_width=90,
    angle_deg=0.0,
) -> np.ndarray:
    """Crée une image synthétique d'une piste avec bordures blanches."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)  # route sombre

    # Dessiner les deux bordures blanches (rectangles verticaux légèrement inclinés)
    offset = int(np.tan(np.radians(angle_deg)) * height / 2)

    left_c  = center_x - track_half_width
    right_c = center_x + track_half_width

    for row in range(height):
        frac = (row - height / 2) / (height / 2)
        shift = int(frac * offset)
        lc = left_c  + shift
        rc = right_c + shift
        if 0 <= lc < width:
            img[row, max(0, lc - 8):min(width, lc + 8)] = (255, 255, 255)
        if 0 <= rc < width:
            img[row, max(0, rc - 8):min(width, rc + 8)] = (255, 255, 255)

    return img


def main():
    detector = LaneDetector(track_width_px=180)

    tests = [
        ("centré",         320,  0.0),
        ("décalé gauche",  260,  0.0),
        ("décalé droite",  380,  0.0),
        ("angle +15°",     320, +15.0),
        ("angle -15°",     320, -15.0),
    ]

    for name, cx, angle in tests:
        img = make_synthetic_track(center_x=cx, angle_deg=angle)
        state = detector.detect(img)
        print(f"[{name:20s}]  d={state.d:+.3f}  theta={state.theta:+.3f}  valid={state.valid}")

        # Optionnel : afficher l'image si X11 disponible
        try:
            vis = draw_debug(img, state)
            cv2.imshow(name, vis)
            cv2.waitKey(800)
        except Exception:
            pass

    cv2.destroyAllWindows()
    print("\nTest terminé.")


if __name__ == "__main__":
    main()
