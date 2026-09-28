#!/usr/bin/env python3
"""
dqn_odom_node.py
================
Nœud ROS2 DQN basé UNIQUEMENT sur /odom.
Aucune caméra, aucun cv_bridge, aucune détection visuelle.

État  : [x_norm, y_norm, yaw_norm, vx_norm, omega_norm]  (dim=5)
Action: 5 commandes (lin, ang) discrètes
Reward: basée sur la distance au rayon de la piste circulaire

Usage :
  ros2 run rl_agent_dqn dqn_odom_node
  ros2 run rl_agent_dqn dqn_odom_node --ros-args -p train_mode:=false -p load_model:=true
"""

import os
import math
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

# Imports du package
from .dqn_model_odom import DQNAgent
from .reward_odom import RewardManager, OdomState


class DQNOdomNode(Node):

    MODEL_SAVE_PATH   = os.path.expanduser("~/ros/dqn_odom_model.pt")
    SAVE_EVERY_STEPS  = 500
    LEARN_EVERY_STEPS = 4
    CONTROL_HZ        = 10.0
    GRACE_STEPS       = 3     # steps sans vérifier done en début d'épisode

    def __init__(self):
        super().__init__("dqn_odom_node")

        # ── Paramètres ──
        self.declare_parameter("train_mode", True)
        self.declare_parameter("load_model", False)
        self.declare_parameter("model_path", self.MODEL_SAVE_PATH)
        self.declare_parameter("control_hz", self.CONTROL_HZ)

        self.train_mode = self.get_parameter("train_mode").value
        load_model      = self.get_parameter("load_model").value
        model_path      = self.get_parameter("model_path").value
        hz              = self.get_parameter("control_hz").value

        # ── Composants ──
        self.agent      = DQNAgent()
        self.reward_mgr = RewardManager()

        if load_model and os.path.exists(model_path):
            self.agent.load(model_path)
            self.get_logger().info(f"Modèle chargé : {model_path}")
        elif load_model:
            self.get_logger().warn(f"Modèle introuvable : {model_path}")

        # Mode évaluation : ε = 0
        if not self.train_mode:
            self.agent.eps_start = 0.0
            self.agent.eps_end   = 0.0

        # ── État ──
        self._odom_state  = OdomState()
        self._odom_ready  = False     # True dès la 1ère réception /odom
        self._prev_state  = None
        self._prev_action = 0

        self._episode_reward = 0.0
        self._episode_steps  = 0
        self._total_steps    = 0
        self._episode        = 0

        # ── ROS ──
        self.create_subscription(Odometry, "/odom", self._cb_odom, 10)
        self._cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_timer(1.0 / hz, self._control_loop)

        # Diagnostic : avertir si /odom absent après 5s
        self._odom_warned = False
        self.create_timer(5.0, self._check_odom)

        self.get_logger().info(
            f"[DQN-Odom] Démarré | "
            f"train={self.train_mode} | "
            f"ε={self.agent.epsilon:.3f}"
        )

    # ──────────────────────────────────────────
    def _cb_odom(self, msg: Odometry):
        """Construit OdomState à chaque message /odom."""
        pos = msg.pose.pose.position
        q   = msg.pose.pose.orientation

        # Extraire yaw depuis le quaternion
        yaw = _quat_to_yaw(q.x, q.y, q.z, q.w)

        vx    = float(msg.twist.twist.linear.x)
        omega = float(msg.twist.twist.angular.z)

        self._odom_state = OdomState(
            x=float(pos.x),
            y=float(pos.y),
            yaw=yaw,
            vx=vx,
            omega=omega,
        )
        self._odom_ready = True

    def _check_odom(self):
        if not self._odom_ready and not self._odom_warned:
            self._odom_warned = True
            self.get_logger().warn(
                "⚠️  Aucun message /odom reçu après 5s !\n"
                "   Vérifiez que le bridge Gazebo est actif :\n"
                "     ros2 topic echo /odom"
            )

    # ──────────────────────────────────────────
    def _control_loop(self):
        # Attendre la première odométrie
        if not self._odom_ready:
            return

        os_  = self._odom_state
        state = os_.to_array()   # shape (5,)

        # ── Sélection action ──
        action = self.agent.select_action(state)
        lin, ang = self.agent.get_cmd(action)
        self._publish_cmd(lin, ang)

        # ── Apprentissage ──
        if self.train_mode and self._prev_state is not None:
            reward = self.reward_mgr.compute(os_)

            in_grace = (self._episode_steps < self.GRACE_STEPS)
            done = (not in_grace) and self.reward_mgr.is_done(os_)

            self.agent.store(
                self._prev_state, self._prev_action,
                reward, state, done,
            )

            if self._total_steps % self.LEARN_EVERY_STEPS == 0:
                loss = self.agent.learn()
                if loss is not None and self._total_steps % 100 == 0:
                    self.get_logger().info(
                        f"ep={self._episode:4d} | "
                        f"step={self._total_steps:6d} | "
                        f"ep_step={self._episode_steps:4d} | "
                        f"ε={self.agent.epsilon:.3f} | "
                        f"loss={loss:.4f} | "
                        f"R={self._episode_reward:.2f} | "
                        f"r={math.hypot(os_.x, os_.y):.3f}m | "
                        f"dist_err={os_.dist_err:+.3f}m"
                    )

            self._episode_reward += reward
            self._episode_steps  += 1

            if done:
                self._reset_episode()

        # ── Sauvegarde ──
        if self.train_mode and self._total_steps > 0 \
                and self._total_steps % self.SAVE_EVERY_STEPS == 0:
            path = self.get_parameter("model_path").value
            self.agent.save(path)
            self.get_logger().info(f"💾 Modèle sauvegardé → {path}")

        self._prev_state  = state.copy()
        self._prev_action = action
        self._total_steps += 1

    # ──────────────────────────────────────────
    def _publish_cmd(self, linear: float, angular: float):
        msg = Twist()
        msg.linear.x  = float(linear)
        msg.angular.z = float(angular)
        self._cmd_pub.publish(msg)

    def _reset_episode(self):
        self.get_logger().info(
            f"=== Episode {self._episode:4d} | "
            f"steps={self._episode_steps:4d} | "
            f"R={self._episode_reward:.2f} | "
            f"r_final={math.hypot(self._odom_state.x, self._odom_state.y):.3f}m ==="
        )
        self._episode        += 1
        self._episode_reward  = 0.0
        self._episode_steps   = 0
        self.reward_mgr.reset()
        self._publish_cmd(0.0, 0.0)


# ──────────────────────────────────────────
def _quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Extrait le yaw (rotation autour de Z) d'un quaternion."""
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


# ──────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = DQNOdomNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Arrêt — sauvegarde du modèle...")
        node.agent.save(node.get_parameter("model_path").value)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
