#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped


class OdomTFPublisher(Node):

    def __init__(self):
        super().__init__('odom_tf_publisher')

        # TF broadcaster
        self.br = TransformBroadcaster(self)

        # Subscribe to /odom
        self.subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.get_logger().info("✅ Odom TF Publisher Started")

    def odom_callback(self, msg):
        t = TransformStamped()

        # Time
        t.header.stamp = self.get_clock().now().to_msg()

        # 🔥 IMPORTANT → utiliser les mêmes noms que /odom
        t.header.frame_id = msg.header.frame_id          # ex: duckiebot/odom
        t.child_frame_id = msg.child_frame_id            # ex: duckiebot/chassis

        # Position
        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z

        # Orientation
        t.transform.rotation = msg.pose.pose.orientation

        # Publish TF
        self.br.sendTransform(t)


def main():
    rclpy.init()
    node = OdomTFPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()