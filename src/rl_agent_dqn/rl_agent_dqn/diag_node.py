#!/usr/bin/env python3
"""
diag_node.py
============
Nœud de diagnostic à lancer AVANT le training pour vérifier :
  • /camera/image_raw  reçu et décodable
  • /odom              reçu
  • LaneDetector       fonctionne sur les vraies images
  • Sauvegarde une image PNG pour inspection visuelle

Usage :
  ros2 run rl_agent_dqn diag_node
  # Ctrl+C après quelques secondes
  # Puis ouvrir ~/ros2_ws/diag_frame.png
"""

import os
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry

try:
    from cv_bridge import CvBridge
    import cv2
    CV_OK = True
except ImportError:
    CV_OK = False

from .lane_detector import LaneDetector, draw_debug


class DiagNode(Node):

    SAVE_PATH = os.path.expanduser("~/ros/diag_frame.png")

    def __init__(self):
        super().__init__("diag_node")

        self.detector    = LaneDetector()
        self.bridge      = CvBridge() if CV_OK else None
        self._img_count  = 0
        self._odom_count = 0
        self._saved      = False

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=1)
        self.create_subscription(Image,    "/camera/image_raw", self._cb_img,  qos)
        self.create_subscription(Odometry, "/odom",             self._cb_odom, 10)
        self.create_timer(2.0, self._report)

        self.get_logger().info("Nœud de diagnostic démarré (attente messages...)")

    def _cb_img(self, msg: Image):
        self._img_count += 1
        if self.bridge is None or not CV_OK:
            return
        try:
            bgr   = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            state = self.detector.detect(bgr)

            if self._img_count % 10 == 0:
                self.get_logger().info(
                    f"  Image #{self._img_count:4d} | "
                    f"shape={bgr.shape} | "
                    f"valid={state.valid} | "
                    f"d={state.d:+.3f} | theta={state.theta:+.3f}"
                )

            # Sauvegarder la 5ème image pour inspection
            if self._img_count == 5 and not self._saved:
                self._saved = True
                vis = draw_debug(bgr, state)
                cv2.imwrite(self.SAVE_PATH, vis)
                self.get_logger().info(f"  📸 Image sauvegardée → {self.SAVE_PATH}")

                # Analyser les couleurs présentes
                hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
                white_mask = cv2.inRange(hsv,
                    np.array([0, 0, 180]), np.array([180, 50, 255]))
                pct_white = 100.0 * np.sum(white_mask > 0) / white_mask.size
                self.get_logger().info(
                    f"  Pixels blancs détectés : {pct_white:.1f}% de l'image\n"
                    f"  (si < 1% → bordures non visibles → ajuster HSV ou position spawn)"
                )

        except Exception as e:
            self.get_logger().error(f"Erreur image : {e}")

    def _cb_odom(self, msg: Odometry):
        self._odom_count += 1
        if self._odom_count == 1:
            pos = msg.pose.pose.position
            self.get_logger().info(
                f"  /odom reçu ✓ | position x={pos.x:.3f} y={pos.y:.3f}"
            )

    def _report(self):
        self.get_logger().info(
            f"--- Rapport diagnostic ---\n"
            f"  /camera/image_raw : {'✓ ' + str(self._img_count) + ' images' if self._img_count > 0 else '✗ AUCUNE IMAGE (bridge image manquant ?)'}\n"
            f"  /odom             : {'✓ ' + str(self._odom_count) + ' msgs'   if self._odom_count > 0 else '✗ AUCUN ODOM (bridge gz absent ?)'}\n"
            f"  cv_bridge         : {'✓' if CV_OK else '✗ NON INSTALLÉ'}\n"
            f"  Image sauvegardée : {self.SAVE_PATH if self._saved else 'non encore'}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = DiagNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
