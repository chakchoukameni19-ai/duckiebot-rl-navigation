#!/usr/bin/env python3
import os
import time
import subprocess
import random
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image

from cv_bridge import CvBridge

from .dqn_model import DQNAgent
from .lane_detector import LaneDetector, LaneState
from .reward import RewardManager


class DQNTrainingNode(Node):

    MODEL_SAVE_PATH = os.path.expanduser("~/ros/models/dqn_model.pt")

    SAVE_EVERY_STEPS = 1000
    LEARN_EVERY_STEPS = 4
    CONTROL_HZ = 10.0
    MAX_STEPS_PER_EP = 1000
    GRACE_STEPS = 5

    def __init__(self):
        super().__init__("dqn_training_node")

        self.agent = DQNAgent()
        self.detector = LaneDetector()
        self.reward_mgr = RewardManager()
        self.bridge = CvBridge()

        # état
        self._lane_state = LaneState()
        self._omega = 0.0
        self._prev_state = None
        self._prev_action = 0

        self._episode_reward = 0.0
        self._episode_steps = 0
        self._total_steps = 0
        self._episode = 0

        # QoS Gazebo
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=1
        )

        # 📷 IMAGE
        self.create_subscription(
            Image,
            "/world/duckietown_world/model/duckiebot/link/chassis/sensor/camera_sensor/image",
            self._cb_image,
            qos
        )

        # 📍 ODOM
        self.create_subscription(
            Odometry,
            "/odom",
            self._cb_odom,
            10
        )

        self._cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        self.create_timer(1.0 / self.CONTROL_HZ, self._control_loop)

        self.get_logger().info("🚀 DQN TRAINING STARTED")

    # =========================
    # CALLBACKS
    # =========================
    def _cb_image(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            self._lane_state = self.detector.detect(img)
        except Exception as e:
            self.get_logger().warn(f"Image error: {e}")

    def _cb_odom(self, msg):
        self._omega = float(msg.twist.twist.angular.z)

    # =========================
    # CONTROL LOOP
    # =========================
    def _control_loop(self):

        ls = self._lane_state

        state = np.array([
            ls.d if ls.valid else 0.0,
            ls.theta if ls.valid else 0.0,
            np.clip(self._omega / 2.0, -1.0, 1.0)
        ], dtype=np.float32)

        action = self.agent.select_action(state)
        lin, ang = self.agent.get_cmd(action)
        self._publish_cmd(lin, ang)

        if self._prev_state is not None:

            reward = self.reward_mgr.compute(ls, self._omega)

            done = (
                self._episode_steps > self.GRACE_STEPS and
                (self.reward_mgr.is_done(ls) or
                 self._episode_steps > self.MAX_STEPS_PER_EP)
            )

            self.agent.store(self._prev_state, self._prev_action, reward, state, done)

            if self._total_steps % self.LEARN_EVERY_STEPS == 0:
                loss = self.agent.learn()

                if loss is not None and self._total_steps % 100 == 0:
                    self.get_logger().info(
                        f"ep={self._episode} step={self._total_steps} "
                        f"ε={self.agent.epsilon:.3f} loss={loss:.4f} "
                        f"R={self._episode_reward:.2f} "
                        f"d={ls.d:.3f} theta={ls.theta:.3f} valid={ls.valid}"
                    )

            self._episode_reward += reward
            self._episode_steps += 1

            if done:
                self._reset_episode()

        # 💾 SAVE MODEL
        if self._total_steps > 0 and self._total_steps % self.SAVE_EVERY_STEPS == 0:
            path = self.MODEL_SAVE_PATH
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.agent.save(path)
            self.get_logger().info(f"💾 Model saved → {path}")

        self._prev_state = state.copy()
        self._prev_action = action
        self._total_steps += 1

    # =========================
    # RESET EPISODE
    # =========================
    def _reset_episode(self):

        self.get_logger().info(
            f"=== EP {self._episode} | R={self._episode_reward:.2f} ==="
        )

        self._episode += 1
        self._episode_reward = 0.0
        self._episode_steps = 0
        self.reward_mgr.reset()

        self._publish_cmd(0.0, 0.0)

        # 🔥 RESET PROPRE (pas de disparition)
        x = random.uniform(-0.2, 0.2)
        y = random.uniform(-0.2, 0.2)

        subprocess.run([
            "gz", "service",
            "-s", "/world/duckietown_world/set_pose",
            "--reqtype", "gz.msgs.Pose",
            "--reptype", "gz.msgs.Boolean",
            "--timeout", "1000",
            "--req",
            'name: "duckiebot", position: {x: 1.8, y: 0.0, z: 0.1}, orientation: {x: 0, y: 0, z: 0, w: 1}'
        ])
        time.sleep(1.0)

    # =========================
    def _publish_cmd(self, lin, ang):
        msg = Twist()
        msg.linear.x = float(lin)
        msg.angular.z = float(ang)
        self._cmd_pub.publish(msg)


def main():
    rclpy.init()
    node = DQNTrainingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()