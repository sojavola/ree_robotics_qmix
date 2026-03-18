#!/usr/bin/env python3
"""
REE Exploration Visualization Launch File
==========================================
Lance tous les nodes de visualisation RViz2 pour le systeme REE.
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """Genere la description du launch"""

    # === ARGUMENTS ===
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Launch RViz2 with visualization config'
    )

    map_width_arg = DeclareLaunchArgument(
        'map_width',
        default_value='100',
        description='Width of the map in cells'
    )

    map_height_arg = DeclareLaunchArgument(
        'map_height',
        default_value='100',
        description='Height of the map in cells'
    )

    num_robots_arg = DeclareLaunchArgument(
        'num_robots',
        default_value='4',
        description='Number of robots'
    )

    cell_size_arg = DeclareLaunchArgument(
        'cell_size',
        default_value='1.0',
        description='Size of each cell in meters'
    )

    viz_rate_arg = DeclareLaunchArgument(
        'visualization_rate',
        default_value='5.0',
        description='Visualization update rate in Hz'
    )

    # === PATHS ===
    pkg_share = FindPackageShare('ree_exploration_viz')
    rviz_config = PathJoinSubstitution([pkg_share, 'config', 'ree_exploration.rviz'])

    # === PARAMETRES COMMUNS ===
    common_params = {
        'map_width': LaunchConfiguration('map_width'),
        'map_height': LaunchConfiguration('map_height'),
        'num_robots': LaunchConfiguration('num_robots'),
        'cell_size': LaunchConfiguration('cell_size'),
    }

    # === NODES ===

    # Node principal de visualisation
    # mineral_threshold=0.30 : affiche uniquement les vrais gisements (pas les queues gaussiennes)
    # show_underground=False, show_veins=False : evite le double rendu
    visualization_node = Node(
        package='ree_exploration_viz',
        executable='visualization_node',
        name='ree_visualization_node',
        output='screen',
        parameters=[
            common_params,
            {'visualization_rate': LaunchConfiguration('visualization_rate')},
            {'mineral_threshold': 0.30},
            {'show_underground': False},
            {'show_veins': False},
            {'show_clusters': True},
        ],
        remappings=[
            ('/mineral_map', '/mineral_map'),
            ('/obstacle_map', '/obstacle_map'),
            ('/underground_layers', '/underground_layers'),
        ]
    )

    # Node de markers robots
    robot_marker_node = Node(
        package='ree_exploration_viz',
        executable='robot_marker_publisher',
        name='robot_marker_publisher',
        output='screen',
        parameters=[
            common_params,
            {'trail_length': 50},
            {'detection_radius': 5.0},
        ]
    )

    # RViz2
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_rviz'))
    )

    # === LAUNCH DESCRIPTION ===
    # NOTE: mineral_heatmap_publisher est desactive — visualization_node gere deja l'affichage
    return LaunchDescription([
        # Fix snap/libpthread conflict for RViz2
        SetEnvironmentVariable('LD_PRELOAD', '/usr/lib/x86_64-linux-gnu/libpthread.so.0'),

        # Arguments
        use_rviz_arg,
        map_width_arg,
        map_height_arg,
        num_robots_arg,
        cell_size_arg,
        viz_rate_arg,

        # Nodes de visualisation
        visualization_node,
        robot_marker_node,

        # RViz (demarre apres 2 secondes pour laisser les nodes s'initialiser)
        TimerAction(
            period=2.0,
            actions=[rviz_node]
        ),
    ])
