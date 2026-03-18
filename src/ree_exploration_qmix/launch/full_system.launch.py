#!/usr/bin/env python3
"""
Launch complet : server + QMIX (trainer + 4 agents) + visualisation
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    qmix_config = os.path.join(
        get_package_share_directory('ree_exploration_qmix'),
        'config', 'qmix_params.yaml'
    )

    viz_launch_dir = os.path.join(
        get_package_share_directory('ree_exploration_viz'),
        'launch'
    )

    # === 1. Serveur REE ===
    server_node = Node(
        package='ree_exploration_server',
        executable='server_node',
        name='ree_exploration_server',
        output='screen'
    )

    # === 2. QMIX Trainer ===
    qmix_trainer = Node(
        package='ree_exploration_qmix',
        executable='qmix_trainer',
        name='qmix_trainer',
        parameters=[{'config_file': qmix_config}],
        output='screen'
    )

    # === 3. QMIX Agents (x4) ===
    agent_nodes = [
        Node(
            package='ree_exploration_qmix',
            executable='qmix_agent',
            name=f'qmix_agent_{i}',
            parameters=[{'config_file': qmix_config}],
            arguments=[str(i)],
            output='screen'
        )
        for i in range(4)
    ]

    # === 4. Visualisation (sans RViz) ===
    viz_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(viz_launch_dir, 'visualization.launch.py')
        ),
        launch_arguments={'use_rviz': 'false'}.items()
    )

    return LaunchDescription([
        # Serveur en premier
        server_node,

        # QMIX demarre apres 2s (laisser le serveur publier les maps)
        TimerAction(period=2.0, actions=[qmix_trainer, *agent_nodes]),

        # Visualisation demarre apres 4s
        TimerAction(period=4.0, actions=[viz_launch]),
    ])
