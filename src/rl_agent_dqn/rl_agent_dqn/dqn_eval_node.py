#!/usr/bin/env python3
"""
dqn_eval_node.py
================
Nœud d'évaluation (exploitation pure) : charge un modèle entraîné
et fait tourner le duckiebot sans exploration (ε = 0).

Usage :
  ros2 run rl_agent_dqn dqn_eval_node --ros-args \
    -p model_path:=/home/user/ros2_ws/dqn_model.pt
"""

import os
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image

try:
    from cv_bridge import CvBridge
    CV_BRIDGE_OK = True
except ImportError:
    CV_BRIDGE_OK = False

from .dqn_model import DQNAgent
from .lane_detector import LaneDetector, LaneState


class DQNEvalNode(Node):

    def __init__(self):
        super().__init__("dqn_eval_node")

        self.declare_parameter("model_path", os.path.expanduser("~/ros/dqn_model.pt"))
        self.declare_parameter("control_hz", 10.0)

        model_path = self.get_parameter("model_path").value
        hz         = self.get_parameter("control_hz").value

        self.agent    = DQNAgent()
        self.detector = LaneDetector()
        self.bridge   = CvBridge() if CV_BRIDGE_OK else None

        if os.path.exists(model_path):
            self.agent.load(model_path)
            self.get_logger().info(f"Modèle chargé : {model_path}")
        else:
            self.get_logger().error(f"Modèle introuvable : {model_path}")

        # Forcer ε = 0 (pure exploitation)
        self.agent.eps_start = 0.0
        self.agent.eps_end   = 0.0

        self._lane_state = LaneState()
        self._omega      = 0.0

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=1)
        self.create_subscription(Image,    "/camera/image_raw", self._cb_image, qos)
        self.create_subscription(Odometry, "/odom",             self._cb_odom,  10)

        self._cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_timer(1.0 / hz, self._control_loop)

        self.get_logger().info("DQN Eval Node démarré (ε=0, exploitation pure)")

    def _cb_image(self, msg):
        if self.bridge:
            try:
                bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                self._lane_state = self.detector.detect(bgr)
            except Exception as e:
                self.get_logger().warn(str(e), throttle_duration_sec=2.0)

    def _cb_odom(self, msg):
        self._omega = float(msg.twist.twist.angular.z)

    def _control_loop(self):
        ls = self._lane_state
        state = np.array([
            ls.d     if ls.valid else 0.0,
            ls.theta if ls.valid else 0.0,
            np.clip(self._omega / 2.0, -1.0, 1.0),
        ], dtype=np.float32)

        action = self.agent.select_action(state)
        lin, ang = self.agent.get_cmd(action)

        msg = Twist()
        msg.linear.x  = lin
        msg.angular.z = ang
        self._cmd_pub.publish(msg)

        self.get_logger().info(
            f"action={action} | lin={lin:.2f} ang={ang:.2f} | "
            f"d={ls.d:.3f} theta={ls.theta:.3f} valid={ls.valid}",
            throttle_duration_sec=0.5
        )


def main(args=None):
    rclpy.init(args=args)
    node = DQNEvalNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
