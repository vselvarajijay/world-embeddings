#!/usr/bin/env python3
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os


def generate_launch_description():
    pkg_share = get_package_share_directory("world_embeddings")
    config_path = os.path.join(pkg_share, "config", "params.yaml")

    return LaunchDescription([
        DeclareLaunchArgument("params_file", default_value=config_path, description="Path to params YAML"),
        Node(
            package="world_embeddings",
            executable="world_embeddings_node",
            name="world_embeddings_node",
            output="screen",
            parameters=[LaunchConfiguration("params_file")],
        ),
    ])
