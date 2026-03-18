#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Chemin vers le fichier de paramètres
    config_file = os.path.join(
        get_package_share_directory('ree_exploration_qmix'),
        'config',
        'qmix_params.yaml'
    )
    
    # Arguments
    num_robots_arg = DeclareLaunchArgument(
        'num_robots',
        default_value='4',
        description='Number of robots'
    )
    
    # Nœud serveur d'exploration (existant)
    server_node = Node(
        package='ree_exploration_agent',
        executable='server_node',
        name='ree_exploration_server',
        output='screen'
    )
    
    # Nœud d'entraînement QMIX
    qmix_trainer = Node(
        package='ree_exploration_qmix',
        executable='qmix_trainer',
        name='qmix_trainer',
        parameters=[config_file],
        output='screen'
    )
    
    # Nœuds agents QMIX
    agent_nodes = []
    for i in range(4):
        agent_node = Node(
            package='ree_exploration_qmix',
            executable='qmix_agent',
            name=f'qmix_agent_{i}',
            parameters=[config_file, {'robot_id': i}],
            arguments=[str(i)],
            output='screen'
        )
        agent_nodes.append(agent_node)
    
    return LaunchDescription([
        num_robots_arg,
        server_node,
        qmix_trainer,
        *agent_nodes
    ])