#!/usr/bin/env python3
"""
Launch file pour executer UNIQUEMENT ree_exploration_qmix
(trainer + 4 agents) sans le server.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    config_file = os.path.join(
        get_package_share_directory('ree_exploration_qmix'),
        'config',
        'qmix_params.yaml'
    )

    qmix_trainer = Node(
        package='ree_exploration_qmix',
        executable='qmix_trainer',
        name='qmix_trainer',
        parameters=[{'config_file': config_file}],
        output='screen'
    )

    agent_nodes = [
        Node(
            package='ree_exploration_qmix',
            executable='qmix_agent',
            name=f'qmix_agent_{i}',
            parameters=[{'config_file': config_file}],
            arguments=[str(i)],
            output='screen'
        )
        for i in range(4)
    ]

    return LaunchDescription([qmix_trainer, *agent_nodes])
