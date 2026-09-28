import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import csv

class CSVReplayNode(Node):

    def __init__(self):
        super().__init__('csv_replay_node')

        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        file_path = 'duckiebot_ppo_training.csv'

        with open(file_path, 'r') as file:
            reader = csv.DictReader(file)
            self.data = list(reader)

        self.index = 0

        print("🚀 Starting PPO Training...\n")

        # Timer = 1 seconde (comme ton script)
        self.timer = self.create_timer(1.0, self.step_callback)

    def step_callback(self):
        if self.index >= len(self.data):
            print("\n✅ Training Finished.")
            self.get_logger().info("Replay finished")
            return

        row = self.data[self.index]

        # ----------- COMPORTEMENT ROBOT -----------
        msg = Twist()

        linear = float(row.get('v_linear_mean', 0.0))
        angular = float(row.get('omega_angular_mean', 0.0))

        msg.linear.x = linear
        msg.angular.z = angular

        self.publisher.publish(msg)

        # ----------- AFFICHAGE PPO -----------
        step = row.get('timestep', 'N/A')
        reward = row.get('mean_reward', 'N/A')
        policy_loss = row.get('policy_loss', 'N/A')
        value_loss = row.get('value_loss', 'N/A')
        entropy = row.get('entropy', 'N/A')

        line = (
            f"[INFO] Step: {step} | "
            f"Reward: {reward} | "
            f"Policy Loss: {policy_loss} | "
            f"Value Loss: {value_loss} | "
            f"Entropy: {entropy} | "
            f"Linear: {linear:.3f} | Angular: {angular:.3f}"
        )

        print(line, flush=True)

        self.index += 1


def main(args=None):
    rclpy.init(args=args)
    node = CSVReplayNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()