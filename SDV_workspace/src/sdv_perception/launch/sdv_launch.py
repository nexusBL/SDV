import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    """Launch completely integrated SDV stack (Perception + Control)."""
    return LaunchDescription([
        Node(
            package='sdv_perception',
            executable='perception_node',
            name='sdv_perception_node',
            output='screen'
        ),
        Node(
            package='sdv_perception',
            executable='perception_node',
            name='sdv_perception_node',
            output='screen'
        ),
        Node(
            package='sdv_perception',
            executable='camera_bridge',
            name='sdv_camera_bridge',
            output='screen'
        ),
        Node(
            package='sdv_control',
            executable='control_node',
            name='sdv_control_node',
            output='screen'
        )
    ])
