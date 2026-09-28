#!/usr/bin/env python3
"""
ppo_train.launch.py
===================
Lance le nœud d'entraînement PPO + bridge Gazebo.
"""

import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    bridge_config = os.path.join(
        get_package_share_directory("rl_agent_ppo"),
        "config",
        "bridge.yaml",
    )

    # Nœud bridge ROS2 ↔ Gazebo
    bridge_node = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="gz_bridge",
        parameters=[{"config_file": bridge_config}],
        output="screen",
    )

    # Nœud d'entraînement PPO (démarre 2 s après le bridge)
    ppo_train_node = TimerAction(
        period=2.0,
        actions=[
            Node(
                package="rl_agent_ppo",
                executable="ppo_training_node",
                name="ppo_training_node",
                output="screen",
                emulate_tty=True,
            )
        ],
    )

    return LaunchDescription([
        bridge_node,
        ppo_train_node,
    ])
