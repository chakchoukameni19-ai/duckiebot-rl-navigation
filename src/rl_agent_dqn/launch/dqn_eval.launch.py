#!/usr/bin/env python3
"""
dqn_eval.launch.py
==================
Lance uniquement le nœud d'évaluation (sans réentraînement).
Gazebo doit déjà être lancé.

Usage :
  ros2 launch rl_agent_dqn dqn_eval.launch.py
  ros2 launch rl_agent_dqn dqn_eval.launch.py model_path:=/tmp/dqn_model.pt
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    model_path_arg = DeclareLaunchArgument(
        "model_path",
        default_value=os.path.expanduser("~/ros2_ws/dqn_model.pt"),
    )

    dqn_eval = Node(
        package="rl_agent_dqn",
        executable="dqn_eval_node",
        name="dqn_eval_node",
        output="screen",
        parameters=[{
            "model_path": LaunchConfiguration("model_path"),
            "control_hz": 10.0,
        }],
    )

    return LaunchDescription([model_path_arg, dqn_eval])
