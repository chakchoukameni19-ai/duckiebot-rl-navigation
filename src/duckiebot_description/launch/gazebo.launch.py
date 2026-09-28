#!/usr/bin/env python3
import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():

    pkg_share = FindPackageShare('duckiebot_description').find('duckiebot_description')
    xacro_file = os.path.join(pkg_share, 'urdf', 'duckiebot.urdf.xacro')
    bridge_config = os.path.join(pkg_share, 'config', 'bridge.yaml')
    world_path = os.path.join(pkg_share, 'worlds', 'empty.sdf')

    robot_description = ParameterValue(
    Command([FindExecutable(name='xacro'), ' ', xacro_file]),
    value_type=str   # ← force le type string, évite l'ambiguïté YAML
)
    rsp = Node(
    package='robot_state_publisher',
    executable='robot_state_publisher',
    parameters=[{'robot_description': robot_description}]
)
    # -----------------------------
    # RViz2
    # -----------------------------
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen'
    )
    # -----------------------------
    # Launch Gazebo Harmonic
    # -----------------------------
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                FindPackageShare('ros_gz_sim').find('ros_gz_sim'),
                'launch',
                'gz_sim.launch.py'
            )
        ),
        launch_arguments={'gz_args': f'-r {world_path}'}.items()
    )
    # -----------------------------
    # Spawn robot
    # -----------------------------
    spawn = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=[
            '-name', 'duckiebot',
            '-topic', 'robot_description',
            '-x', '1.8',
            '-y', '0.0',
            '-z', '0.1'
        ]
    )
    # -----------------------------
    # ROS <-> Gazebo Bridge
    # -----------------------------
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        parameters=[{'config_file': bridge_config}],
        output='screen'
    )
    return LaunchDescription([
        rsp,
        rviz,
        gazebo,
        spawn,
        bridge,
    ])