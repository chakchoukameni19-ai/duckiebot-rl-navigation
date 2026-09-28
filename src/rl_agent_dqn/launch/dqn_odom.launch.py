#!/usr/bin/env python3
"""
dqn_odom.launch.py
==================
Lance Gazebo + robot_state_publisher + bridge + nœud DQN odom-only.

Usage :
  ros2 launch rl_agent_dqn dqn_odom.launch.py
  ros2 launch rl_agent_dqn dqn_odom.launch.py train_mode:=false load_model:=true
"""

import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command, FindExecutable
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():

    pkg_duck = FindPackageShare("duckiebot_description").find("duckiebot_description")

    xacro_file = os.path.join(pkg_duck, "urdf",   "duckiebot.urdf.xacro")
    bridge_cfg = os.path.join(pkg_duck, "config", "bridge.yaml")
    world_path = os.path.join(pkg_duck, "worlds", "empty.sdf")
    model_path = os.path.expanduser("~/ros2_ws/dqn_odom_model.pt")

    # ── Arguments ──
    train_arg = DeclareLaunchArgument("train_mode", default_value="true")
    load_arg  = DeclareLaunchArgument("load_model", default_value="false")

    # ── robot_state_publisher ──
    robot_desc = ParameterValue(
        Command([FindExecutable(name="xacro"), " ", xacro_file]),
        value_type=str,
    )
    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description": robot_desc}],
    )

    # ── Gazebo ──
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                FindPackageShare("ros_gz_sim").find("ros_gz_sim"),
                "launch", "gz_sim.launch.py",
            )
        ),
        launch_arguments={"gz_args": f"-r {world_path}"}.items(),
    )

    # ── Spawn ──
    spawn = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-name",  "duckiebot",
            "-topic", "robot_description",
            "-x", "1.8", "-y", "0.0", "-z", "0.1",
        ],
    )

    # ── Bridge GZ ↔ ROS2 ──
    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        parameters=[{"config_file": bridge_cfg}],
        output="screen",
    )

    # ── Nœud DQN odom-only ──
    dqn = Node(
        package="rl_agent_dqn",
        executable="dqn_odom_node",
        name="dqn_odom_node",
        output="screen",
        parameters=[{
            "train_mode": LaunchConfiguration("train_mode"),
            "load_model": LaunchConfiguration("load_model"),
            "model_path": model_path,
            "control_hz": 10.0,
        }],
    )

    return LaunchDescription([
        train_arg,
        load_arg,
        rsp,
        gazebo,
        spawn,
        bridge,
        dqn,
    ])
