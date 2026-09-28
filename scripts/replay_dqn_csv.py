import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import csv
import math

class DQNReplaySteps(Node):

    def __init__(self):
        super().__init__('dqn_steps_node')

        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        file_path = 'dqn_steps_2episodes.csv'

        with open(file_path, 'r') as file:
            reader = csv.DictReader(file)
            self.data = list(reader)

        self.index = 0
        self.current_episode = None

        print("🚀 DQN Step-by-Step Simulation (REALISTIC MOTION)...\n")

        # plus rapide → mouvement fluide
        self.timer = self.create_timer(0.1, self.step_callback)

    def step_callback(self):
        if self.index >= len(self.data):
            print("\n✅ Simulation Finished")
            return

        row = self.data[self.index]

        episode = row['episode']
        step = int(row['step'])
        reward = float(row['reward'])

        # ----------- NOUVEL EPISODE -----------
        if episode != self.current_episode:
            print(f"\n🔥 Episode {episode} START\n")
            self.current_episode = episode

        msg = Twist()

        # ----------- CONTRÔLE RÉALISTE -----------

        # 🔥 vitesse réduite (très important)
        linear = 0.15

        # 🔥 créer une trajectoire circulaire (anneau)
        base_curve = math.sin(step * 0.15)

        # 🔥 amplification forte pour éviter ligne droite
        angular = 1.2 * base_curve

        # 🔥 adaptation selon reward (comportement RL)
        if reward > 40:
            angular *= 0.5
        elif reward > 10:
            angular *= 0.8
        elif reward > -10:
            angular *= 1.2
        else:
            angular *= 1.8

        # 🔥 sécurité anti ligne droite
        if abs(angular) < 0.2:
            angular = 0.5 if (step % 2 == 0) else -0.5

        # 🔥 limiter valeurs
        angular = max(-2.0, min(2.0, angular))

        msg.linear.x = linear
        msg.angular.z = angular

        self.publisher.publish(msg)

        # ----------- LOG -----------

        print(
            f"[Ep {episode} | Step {step}] "
            f"Reward: {row['reward']} | "
            f"CumReward: {row['cumulative_reward']} | "
            f"Loss: {row['loss']} | "
            f"Epsilon: {row['epsilon']} | "
            f"Lin: {linear:.2f} | Ang: {angular:.2f}",
            flush=True
        )

        self.index += 1


def main(args=None):
    rclpy.init(args=args)
    node = DQNReplaySteps()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()