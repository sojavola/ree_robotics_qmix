#!/usr/bin/env python3
"""
Full REE Exploration System Launch File
=======================================
Lance le serveur d'exploration ET la visualisation RViz2.
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Genere la description du launch complet"""

    # === ARGUMENTS ===
    use_advanced_arg = DeclareLaunchArgument(
        'use_advanced_generator',
        default_value='true',
        description='Use advanced geological mineral generator'
    )

    map_width_arg = DeclareLaunchArgument(
        'map_width',
        default_value='100',
        description='Width of the map'
    )

    map_height_arg = DeclareLaunchArgument(
        'map_height',
        default_value='100',
        description='Height of the map'
    )

    num_robots_arg = DeclareLaunchArgument(
        'num_robots',
        default_value='4',
        description='Number of robots'
    )

    # === NODES ===

    # Serveur d'exploration REE
    server_node = Node(
        package='ree_exploration_server',
        executable='server_node',
        name='ree_exploration_server',
        output='screen',
        parameters=[
            {'use_advanced_generator': LaunchConfiguration('use_advanced_generator')},
        ]
    )

    # Visualization launch (inclus apres un delai)
    viz_pkg_share = FindPackageShare('ree_exploration_viz')
    viz_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([viz_pkg_share, 'launch', 'visualization.launch.py'])
        ]),
        launch_arguments={
            'map_width': LaunchConfiguration('map_width'),
            'map_height': LaunchConfiguration('map_height'),
            'num_robots': LaunchConfiguration('num_robots'),
            'use_rviz': 'true',
        }.items()
    )

    # === LAUNCH DESCRIPTION ===
    return LaunchDescription([
        # Arguments
        use_advanced_arg,
        map_width_arg,
        map_height_arg,
        num_robots_arg,

        # Serveur (demarre en premier)
        server_node,

        # Visualisation (demarre apres 3 secondes)
        TimerAction(
            period=3.0,
            actions=[viz_launch]
        ),
    ])
