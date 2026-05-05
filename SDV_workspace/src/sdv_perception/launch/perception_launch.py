#!/usr/bin/env python3
"""
SDV Perception Pipeline — ROS2 Launch File.
Starts the perception node with appropriate parameters.
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='sdv_perception',
            executable='perception_node',
            name='sdv_perception',
            output='screen',
            emulate_tty=True,
            parameters=[],
        ),
    ])
