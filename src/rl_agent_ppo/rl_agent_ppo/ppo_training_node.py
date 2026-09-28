#!/usr/bin/env python3
"""
ppo_training_node.py
====================
Nœud ROS2 d'entraînement PPO pour le suivi de piste circulaire (Duckiebot).

COMPORTEMENT :
  - Reward cumulé entre -250 et +100
  - Reset automatique si reward cumulé ≤ -250
  - Reset automatique après 2000 steps sur piste (bon épisode)
  - Après chaque reset → position aléatoire SUR la piste
  - Affiche et enregistre les paramètres PPO à chaque step et après chaque épisode
  - Mise à jour PPO tous les ROLLOUT_LEN=256 steps
"""

import os
import csv
import time
import random
import subprocess
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image

from cv_bridge import CvBridge

from .ppo_model import PPOAgent
from .lane_detector import LaneDetector, LaneState
from .ppo_reward import PPORewardManager


# ────────────────────────────────────────────
# Positions aléatoires sur la piste circulaire
# (à adapter selon votre carte Gazebo)
# ────────────────────────────────────────────
TRACK_SPAWN_POSITIONS = [
    # (x, y, yaw_deg)  — yaw en degrés, sera converti en quaternion
    ( 1.80,  0.00,   0),
    ( 1.27,  1.27,  45),
    ( 0.00,  1.80,  90),
    (-1.27,  1.27, 135),
    (-1.80,  0.00, 180),
    (-1.27, -1.27, 225),
    ( 0.00, -1.80, 270),
    ( 1.27, -1.27, 315),
]


def yaw_to_quat(yaw_deg: float):
    """Convertit un angle yaw (degrés) en quaternion (x,y,z,w)."""
    half = np.radians(yaw_deg) / 2.0
    return 0.0, 0.0, np.sin(half), np.cos(half)


class PPOTrainingNode(Node):

    MODEL_SAVE_PATH  = os.path.expanduser("~/ros/models/ppo_model.pt")
    CSV_STEP_PATH    = os.path.expanduser("~/ros/logs/ppo_steps.csv")
    CSV_EPISODE_PATH = os.path.expanduser("~/ros/logs/ppo_episodes.csv")

    SAVE_EVERY_STEPS = 1000
    CONTROL_HZ       = 10.0
    GRACE_STEPS      = 5      # Pas de reset avant N steps après spawn

    def __init__(self):
        super().__init__("ppo_training_node")

        self.agent      = PPOAgent(rollout_len=256)
        self.detector   = LaneDetector()
        self.reward_mgr = PPORewardManager()
        self.bridge     = CvBridge()

        # État courant
        self._lane_state  = LaneState()
        self._omega       = 0.0
        self._prev_state  = None
        self._prev_action = np.array([0.0, 0.0])
        self._prev_logp   = 0.0
        self._prev_value  = 0.0

        # Compteurs
        self._episode_steps  = 0
        self._total_steps    = 0
        self._episode        = 0
        self._episode_reward = 0.0

        # Métriques PPO (dernière update)
        self._ppo_metrics: dict = {}

        # Init logs CSV
        self._init_csv_logs()

        # QoS Gazebo
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=1)

        # Souscriptions
        self.create_subscription(
            Image,
            "/world/duckietown_world/model/duckiebot/link/chassis/sensor/camera_sensor/image",
            self._cb_image,
            qos,
        )
        

        self._cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        self.create_timer(1.0 / self.CONTROL_HZ, self._control_loop)

        self.get_logger().info("🚀 PPO TRAINING STARTED")

        self.create_subscription(Odometry, "/odom", self._cb_odom, 10)


    def _cb_odom(self, msg):
        self._omega = float(msg.twist.twist.angular.z)

    def _cb_image(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            self._lane_state = self.detector.detect(img)
        except Exception as e:
            self.get_logger().warn(f"Image error: {e}")

    

    # ════════════════════════════════════════
    # BOUCLE DE CONTRÔLE
    # ════════════════════════════════════════

    def _control_loop(self):
        ls = self._lane_state

        # ── État courant ──
        state = np.array([
            ls.d     if ls.valid else 0.0,
            ls.theta if ls.valid else 0.0,
            float(np.clip(self._omega / 2.0, -1.0, 1.0)),
        ], dtype=np.float32)

        # ── Sélection d'action ──
        action, log_prob, value = self.agent.select_action(state)
        lin_vel = float(action[0])
        ang_vel = float(action[1])
        self._publish_cmd(lin_vel, ang_vel)

        # ── Calcul du reward et stockage transition ──
        if self._prev_state is not None:
            reward = self.reward_mgr.compute(ls, self._omega)
            self._episode_reward += reward
            self._episode_steps  += 1
            self._total_steps    += 1

            # Condition done
            done = (
                self._episode_steps > self.GRACE_STEPS and
                self.reward_mgr.is_done(ls)
            )

            # Stocker dans le buffer PPO
            self.agent.store(
                self._prev_state,
                self._prev_action,
                reward,
                float(done),
                self._prev_logp,
                self._prev_value,
            )

            # ── Mise à jour PPO si buffer plein ──
            if self.agent.buffer_ready():
                self._ppo_metrics = self.agent.learn(state)
                self._log_ppo_update()

            # ── Logging step ──
            self._log_step(state, reward, done, ls)

            # ── Reset épisode ──
            if done:
                self._reset_episode()

        # ── Sauvegarde périodique du modèle ──
        if self._total_steps > 0 and self._total_steps % self.SAVE_EVERY_STEPS == 0:
            self._save_model()

        self._prev_state  = state.copy()
        self._prev_action = action.copy()
        self._prev_logp   = log_prob
        self._prev_value  = value

    # ════════════════════════════════════════
    # RESET ÉPISODE
    # ════════════════════════════════════════

    def _reset_episode(self):
        reason = "cumulative≤-250" if self.reward_mgr.cumulative_reward <= -250 else "2000 steps"

        self.get_logger().info(
            f"=== EP {self._episode} END | "
            f"steps={self._episode_steps} | "
            f"R_total={self._episode_reward:.2f} | "
            f"R_cumul={self.reward_mgr.cumulative_reward:.2f} | "
            f"reason={reason} ==="
        )

        # Logger l'épisode
        self._log_episode()

        self._episode       += 1
        self._episode_reward = 0.0
        self._episode_steps  = 0
        self.reward_mgr.reset()

        self._publish_cmd(0.0, 0.0)

        # Position aléatoire sur la piste
        pos = random.choice(TRACK_SPAWN_POSITIONS)
        x, y, yaw_deg = pos
        qx, qy, qz, qw = yaw_to_quat(yaw_deg)

        # Petit offset aléatoire pour varier
        x += random.uniform(-0.05, 0.05)
        y += random.uniform(-0.05, 0.05)

        subprocess.run([
            "gz", "service",
            "-s", "/world/duckietown_world/set_pose",
            "--reqtype", "gz.msgs.Pose",
            "--reptype", "gz.msgs.Boolean",
            "--timeout", "1000",
            "--req",
            (
                f'name: "duckiebot", '
                f'position: {{x: {x:.3f}, y: {y:.3f}, z: 0.1}}, '
                f'orientation: {{x: {qx:.4f}, y: {qy:.4f}, z: {qz:.4f}, w: {qw:.4f}}}'
            ),
        ])

        self.get_logger().info(
            f"🔄 EP {self._episode} | Spawn → x={x:.2f} y={y:.2f} yaw={yaw_deg}°"
        )
        time.sleep(1.0)

    # ════════════════════════════════════════
    # LOGGING
    # ════════════════════════════════════════

    def _init_csv_logs(self):
        """Crée les dossiers et fichiers CSV de log."""
        for path in [self.CSV_STEP_PATH, self.CSV_EPISODE_PATH, self.MODEL_SAVE_PATH]:
            os.makedirs(os.path.dirname(path), exist_ok=True)

        # CSV steps : entête
        if not os.path.exists(self.CSV_STEP_PATH):
            with open(self.CSV_STEP_PATH, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "total_step", "episode", "ep_step",
                    "d", "theta", "omega",
                    "lin_vel", "ang_vel",
                    "reward", "cumul_reward",
                    "on_track", "valid", "both_borders",
                    "done",
                    "policy_loss", "value_loss", "entropy", "total_loss", "update_count",
                ])

        # CSV épisodes : entête
        if not os.path.exists(self.CSV_EPISODE_PATH):
            with open(self.CSV_EPISODE_PATH, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "episode", "total_steps",
                    "ep_steps", "ep_reward", "cumul_reward_at_end",
                    "policy_loss", "value_loss", "entropy", "total_loss", "update_count",
                ])

    def _log_step(self, state, reward, done, ls: LaneState):
        """Affiche dans le terminal et écrit dans le CSV step."""
        m = self._ppo_metrics

        # Terminal : uniquement tous les 50 steps pour ne pas spammer
        if self._total_steps % 50 == 0:
            self.get_logger().info(
                f"[STEP {self._total_steps:6d}] "
                f"ep={self._episode} ep_step={self._episode_steps} | "
                f"d={state[0]:.3f} theta={state[1]:.3f} ω={state[2]:.3f} | "
                f"r={reward:+.3f} R_cumul={self.reward_mgr.cumulative_reward:+.1f} | "
                f"on_track={ls.on_track} valid={ls.valid} | "
                f"policy_loss={m.get('policy_loss', 0):.4f} "
                f"value_loss={m.get('value_loss', 0):.4f} "
                f"entropy={m.get('entropy', 0):.4f} "
                f"updates={m.get('update_count', 0)}"
            )

        # CSV : TOUS les steps
        with open(self.CSV_STEP_PATH, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                self._total_steps,
                self._episode,
                self._episode_steps,
                round(state[0], 4),
                round(state[1], 4),
                round(state[2], 4),
                round(self._prev_action[0], 4),
                round(self._prev_action[1], 4),
                round(reward, 4),
                round(self.reward_mgr.cumulative_reward, 2),
                int(ls.on_track),
                int(ls.valid),
                int(ls.both_borders),
                int(done),
                round(m.get("policy_loss", 0), 6),
                round(m.get("value_loss",  0), 6),
                round(m.get("entropy",      0), 6),
                round(m.get("total_loss",   0), 6),
                m.get("update_count", 0),
            ])

    def _log_episode(self):
        """Affiche et enregistre le résumé de l'épisode terminé."""
        m = self._ppo_metrics

        self.get_logger().info(
            f"📊 EPISODE {self._episode} SUMMARY | "
            f"steps={self._episode_steps} | "
            f"R={self._episode_reward:.2f} | "
            f"R_cumul={self.reward_mgr.cumulative_reward:.2f} | "
            f"policy_loss={m.get('policy_loss', 0):.4f} | "
            f"value_loss={m.get('value_loss', 0):.4f} | "
            f"entropy={m.get('entropy', 0):.4f} | "
            f"updates={m.get('update_count', 0)}"
        )

        with open(self.CSV_EPISODE_PATH, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                self._episode,
                self._total_steps,
                self._episode_steps,
                round(self._episode_reward, 2),
                round(self.reward_mgr.cumulative_reward, 2),
                round(m.get("policy_loss", 0), 6),
                round(m.get("value_loss",  0), 6),
                round(m.get("entropy",      0), 6),
                round(m.get("total_loss",   0), 6),
                m.get("update_count", 0),
            ])

    def _log_ppo_update(self):
        """Affiche les métriques d'une mise à jour PPO."""
        m = self._ppo_metrics
        self.get_logger().info(
            f"🔧 PPO UPDATE #{m['update_count']} | "
            f"policy_loss={m['policy_loss']:.4f} | "
            f"value_loss={m['value_loss']:.4f} | "
            f"entropy={m['entropy']:.4f} | "
            f"total_loss={m['total_loss']:.4f}"
        )

    def _save_model(self):
        self.agent.save(self.MODEL_SAVE_PATH)
        self.get_logger().info(f"💾 Model saved → {self.MODEL_SAVE_PATH}")

    # ════════════════════════════════════════
    # PUBLICATION
    # ════════════════════════════════════════

    def _publish_cmd(self, lin: float, ang: float):
        msg = Twist()
        msg.linear.x  = float(lin)
        msg.angular.z = float(ang)
        self._cmd_pub.publish(msg)


# ════════════════════════════════════════
# MAIN
# ════════════════════════════════════════

def main():
    rclpy.init()
    node = PPOTrainingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("⛔ Training stopped by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
