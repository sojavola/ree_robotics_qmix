#!/usr/bin/env python3

import yaml
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class QMIXConfig:
    """Configuration pour QMIX"""
    
    # Paramètres d'entraînement
    gamma: float = 0.995  # 0.99 → trop myope pour 300 pas (γ^300≈0.05%); 0.995 → γ^300≈22%
    learning_rate: float = 0.0005
    buffer_size: int = 5000
    batch_size: int = 8
    target_update_freq: int = 100
    train_freq: int = 10
    grad_clip: float = 10.0
    n_steps: int = 5  # TD(n) : horizon de retour (5 = bon compromis biais/variance)
    
    # Paramètres d'exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: int = 20000
    
    # Architecture réseau
    hidden_dim: int = 64
    hyper_dim: int = 64
    local_obs_size: int = 20  # Taille de la fenêtre d'observation locale (20×20)

    # Dimensions de l'environnement
    map_width: int = 100
    map_height: int = 100
    num_actions: int = 8
    num_robots: int = 4
    state_channels: int = 6  # 4 minéraux + 1 obstacles + 1 exploration
    penalty_collision: float = -5.0
    
    # Topics ROS
    mineral_map_topic: str = "/mineral_map"
    obstacle_map_topic: str = "/obstacle_map"
    agent_exp_topic: str = "/agent_experience"
    trainer_update_topic: str = "/qmix/weight_update"
    trainer_epsilon_topic: str = "/qmix/epsilon"
    
    robot_id: Optional[int] = None
    
    @classmethod
    def from_yaml(cls, yaml_path: str, robot_id: Optional[int] = None):
        """Charge la configuration depuis un fichier YAML"""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        qmix_params = data.get('qmix', {})
        
        # Extraire les paramètres
        config = cls(
            gamma=qmix_params.get('gamma', 0.99),
            learning_rate=qmix_params.get('learning_rate', 0.0005),
            buffer_size=qmix_params.get('buffer_size', 5000),
            batch_size=qmix_params.get('batch_size', 32),
            target_update_freq=qmix_params.get('target_update_freq', 100),
            train_freq=qmix_params.get('train_freq', 10),
            grad_clip=qmix_params.get('grad_clip', 10.0),
            epsilon_start=qmix_params.get('epsilon_start', 1.0),
            epsilon_end=qmix_params.get('epsilon_end', 0.05),
            epsilon_decay=qmix_params.get('epsilon_decay', 20000),
            n_steps=qmix_params.get('n_steps', 5),
            hidden_dim=qmix_params.get('hidden_dim', 64),
            hyper_dim=qmix_params.get('hyper_dim', 64),
            local_obs_size=qmix_params.get('local_obs_size', 20),
            map_width=qmix_params.get('map_width', 100),
            map_height=qmix_params.get('map_height', 100),
            num_actions=qmix_params.get('num_actions', 8),
            num_robots=qmix_params.get('num_robots', 4)
        )
        
        # Topics ROS
        topics = qmix_params.get('topics', {})
        config.mineral_map_topic = topics.get('mineral_map', "/mineral_map")
        config.obstacle_map_topic = topics.get('obstacle_map', "/obstacle_map")
        config.agent_exp_topic = topics.get('agent_exp', "/agent_experience")
        config.trainer_update_topic = topics.get('trainer_update', "/qmix/weight_update")
        config.trainer_epsilon_topic = topics.get('trainer_epsilon', "/qmix/epsilon")
        
        config.robot_id = robot_id
        
        return config
    
    @property
    def state_shape(self):
        """Forme de l'état (C, H, W)"""
        return (self.state_channels, self.map_height, self.map_width)