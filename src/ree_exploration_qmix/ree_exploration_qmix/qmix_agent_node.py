#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String, Int32, Float32
from geometry_msgs.msg import Pose2D, Twist
from nav_msgs.msg import OccupancyGrid
import numpy as np
import torch
import json
import random
import os
import sys
import re
import time

from .networks import QMixLocalNetwork
from .config import QMIXConfig
from .science_reward_system import RealMineralRewardSystem

class QMIXAgentNode(Node):
    """Nœud agent QMIX décentralisé"""
    
    def __init__(self, robot_id=0):
        super().__init__(f'qmix_agent_{robot_id}')
        
        self.robot_id = robot_id
        self.get_logger().info(f'🎯 Robot ID: {robot_id}')
        
        # Charger la configuration
        self.declare_parameter('config_file', '')
        config_file = self.get_parameter('config_file').get_parameter_value().string_value
        
        if config_file:
            self.config = QMIXConfig.from_yaml(config_file, robot_id)
        else:
            self.config = QMIXConfig(robot_id=robot_id)
        
        # Dimensions
        self.map_width = self.config.map_width
        self.map_height = self.config.map_height
        self.num_actions = self.config.num_actions
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Cartes
        self.mineral_map    = np.zeros((self.map_height, self.map_width, 4), dtype=np.float32)
        self.obstacle_map   = np.zeros((self.map_height, self.map_width), dtype=np.float32)
        self.exploration_map = np.zeros((self.map_height, self.map_width), dtype=np.float32)
        self.global_state   = None
        
        # Position
        self.current_position = self.get_initial_position()
        self.last_position = self.current_position
        
        # Système de récompenses (grid_size = (height, width) comme attendu)
        self.reward_system = RealMineralRewardSystem(
            grid_size=(self.map_height, self.map_width),
            robot_id=robot_id
        )
        self.visited_positions = set()
        self.steps_without_mineral = 0
        
        # Réseau local QMIX
        self._init_local_network()
        
        # Exploration
        self.epsilon = self.config.epsilon_start
        self.steps = 0
        self.episode_reward = 0
        self.total_reward = 0
        self.minerals_collected = 0
        
        # Publishers
        self.position_pub = self.create_publisher(
            Pose2D, f'/robot_{robot_id}/position', 10
        )
        self.velocity_pub = self.create_publisher(
            Twist, f'/robot_{robot_id}/cmd_vel', 10
        )
        self.cleaning_pub = self.create_publisher(
            Float32MultiArray, f'/robot_{robot_id}/cleaning_action', 10
        )
        self.discovery_pub = self.create_publisher(
            Float32MultiArray, '/shared_discoveries', 10
        )
        self.experience_pub = self.create_publisher(
            String, self.config.agent_exp_topic, 10
        )
        self.status_pub = self.create_publisher(
            String, f'/robot_{robot_id}/status', 10
        )
        
        # Subscribers
        self.mineral_sub = self.create_subscription(
            Float32MultiArray, self.config.mineral_map_topic,
            self.mineral_callback, 10
        )
        self.obstacle_sub = self.create_subscription(
            OccupancyGrid, self.config.obstacle_map_topic,
            self.obstacle_callback, 10
        )
        self.weight_sub = self.create_subscription(
            String, self.config.trainer_update_topic,
            self.weight_update_callback, 10
        )
        self.epsilon_sub = self.create_subscription(
            Float32, self.config.trainer_epsilon_topic,
            self.epsilon_callback, 10
        )
        
        # Timers
        self.decision_timer = self.create_timer(1.0, self.make_decision)
        self.status_timer = self.create_timer(5.0, self.publish_status)
        self.position_timer = self.create_timer(0.5, self.publish_position)
        
        self.get_logger().info(f'🚀 QMIX Agent Node {robot_id} initialized')
    
    def _init_local_network(self):
        """Initialise le réseau local"""
        self.local_network = QMixLocalNetwork(
            input_shape=self.config.state_shape,
            num_actions=self.num_actions,
            hidden_dim=self.config.hidden_dim,
            local_obs_size=self.config.local_obs_size
        ).to(self.device)
        
        self.get_logger().info('🧠 Réseau local QMIX initialisé')
    
    def _build_global_state(self):
        """Construit l'état global (6, H, W) : 4 minéraux + obstacles + exploration"""
        mineral_ch   = self.mineral_map.transpose(2, 0, 1)           # (4, H, W)
        obstacle_ch  = (self.obstacle_map > 0).astype(np.float32)[np.newaxis]  # (1, H, W)
        explore_ch   = self.exploration_map[np.newaxis]               # (1, H, W)
        self.global_state = np.concatenate(
            [mineral_ch, obstacle_ch, explore_ch], axis=0
        )  # (6, H, W)

    def mineral_callback(self, msg: Float32MultiArray):
        """Callback pour la carte minérale"""
        data = np.array(msg.data, dtype=np.float32)
        if data.size == self.map_height * self.map_width * 4:
            self.mineral_map = data.reshape(self.map_height, self.map_width, 4)
            self._build_global_state()

    def obstacle_callback(self, msg: OccupancyGrid):
        """Callback pour la carte d'obstacles"""
        data = np.array(msg.data, dtype=np.float32)
        if data.size == self.map_height * self.map_width:
            self.obstacle_map = data.reshape(self.map_height, self.map_width)
            self._build_global_state()
    
    def weight_update_callback(self, msg: String):
        """Callback pour les mises à jour de poids du réseau"""
        try:
            data = json.loads(msg.data)
            if data['type'] == 'weight_update':
                path = data['path']
                checkpoint = torch.load(path, map_location=self.device)
                
                # Réseau partagé → extraire les poids 'shared_agent_network.*'
                state_dict = checkpoint['state_dict']
                local_state_dict = {
                    key.replace('shared_agent_network.', ''): value
                    for key, value in state_dict.items()
                    if key.startswith('shared_agent_network.')
                }
                self.local_network.load_state_dict(local_state_dict)
                self.get_logger().debug(f'🔄 Poids mis à jour (step {data["train_step"]})')
                
        except Exception as e:
            self.get_logger().error(f'❌ Error updating weights: {e}')
    
    def epsilon_callback(self, msg: Float32):
        """Callback pour l'epsilon global"""
        self.epsilon = float(msg.data)
    
    def get_initial_position(self):
        """Retourne une position initiale aléatoire"""
        return (random.randint(5, self.map_width - 6),
                random.randint(5, self.map_height - 6))
    
    def get_local_observation(self):
        """Retourne l'observation locale (6 canaux : 4 minéraux + obstacles + exploration)"""
        x, y = self.current_position

        window_size = self.config.local_obs_size
        half = window_size // 2

        x_min = max(0, x - half)
        x_max = min(self.map_width, x + half)
        y_min = max(0, y - half)
        y_max = min(self.map_height, y + half)

        h = y_max - y_min
        w = x_max - x_min

        # 6 canaux : 4 minéraux + obstacles + exploration
        local_map = np.zeros((window_size, window_size, 6), dtype=np.float32)
        local_map[:h, :w, 0:4] = self.mineral_map[y_min:y_max, x_min:x_max]
        local_map[:h, :w, 4]   = (self.obstacle_map[y_min:y_max, x_min:x_max] > 0).astype(np.float32)
        local_map[:h, :w, 5]   = self.exploration_map[y_min:y_max, x_min:x_max]

        norm_pos = np.array([x / self.map_width, y / self.map_height], dtype=np.float32)
        return local_map, norm_pos
    
    def choose_action(self):
        """Choisit une action selon epsilon-greedy"""
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        
        try:
            with torch.no_grad():
                local_map, norm_pos = self.get_local_observation()
                
                # Convertir en tenseurs
                map_tensor = torch.FloatTensor(local_map).permute(2, 0, 1).unsqueeze(0).to(self.device)
                pos_tensor = torch.FloatTensor(norm_pos).unsqueeze(0).to(self.device)
                
                q_values = self.local_network(map_tensor, pos_tensor)
                return q_values.squeeze().argmax().item()
                
        except Exception as e:
            self.get_logger().error(f'❌ Action selection error: {e}')
            return random.randint(0, self.num_actions - 1)
    
    def make_decision(self):
        """Prend une décision de mouvement"""
        if np.max(self.mineral_map) == 0:
            return
        
        # Choisir une action
        action = self.choose_action()
        
        # Exécuter
        reward, done = self.execute_action(action)
        
        # Mettre à jour les statistiques
        self.episode_reward += reward
        self.total_reward += reward
        self.steps += 1
        
        # Envoyer l'expérience au trainer
        self.publish_experience(action, reward, done)

        # Réinitialiser l'épisode si terminé
        if done:
            self.reward_system.reset_episode()
            self.episode_reward = 0
            self.steps = 0
            self.minerals_collected = 0
            self.visited_positions = set()
            self.exploration_map = np.zeros((self.map_height, self.map_width), dtype=np.float32)

        # Log périodique
        if self.steps % 10 == 0:
            self.get_logger().debug(
                f'Step {self.steps}: Action {action}, '
                f'Reward {reward:.2f}, ε={self.epsilon:.3f}'
            )
    
    def execute_action(self, action):
        """Exécute une action"""
        x, y = self.current_position
        
        # Vecteurs de déplacement
        direction_vectors = [
            (0, 1), (0, -1), (-1, 0), (1, 0),
            (-1, 1), (1, 1), (-1, -1), (1, -1)
        ]
        
        dx, dy = direction_vectors[action % len(direction_vectors)]
        new_x = max(0, min(self.map_width - 1, x + dx))
        new_y = max(0, min(self.map_height - 1, y + dy))
        
        if self.is_valid_position(new_x, new_y):
            self.last_position = self.current_position
            self.current_position = (new_x, new_y)
            # Marquer la cellule comme visitée dans la carte d'exploration
            self.exploration_map[new_y, new_x] = 1.0
            reward = self.calculate_reward(new_x, new_y)
            done = self.is_episode_done()
            self.publish_cleaning_action(new_x, new_y)
            
            if reward > 50.0:
                self.publish_discovery(new_x, new_y)
        else:
            reward = self.config.penalty_collision
            done = False
        
        self.publish_velocity()
        return reward, done
    
    def is_valid_position(self, x, y):
        """Vérifie si la position est valide"""
        return (0 <= x < self.map_width and 
                0 <= y < self.map_height and 
                int(self.obstacle_map[y, x]) == 0)
    
    def calculate_reward(self, x, y):
        """Calcule la récompense"""
        try:
            mineral_concentrations = self.mineral_map[y, x, :].tolist()
            position_key = (int(x), int(y))
            is_new_position = position_key not in self.visited_positions
            has_collision = int(self.obstacle_map[y, x]) != 0
            
            reward = self.reward_system.calculate_reward(
                mineral_concentrations=mineral_concentrations,
                position=(x, y),
                is_new_position=is_new_position,
                has_collision=has_collision,
                step_count=self.steps
            )
            
            self.visited_positions.add(position_key)
            
            if reward > 50.0:
                max_concentration = max(mineral_concentrations) if mineral_concentrations else 0.0
                mineral_type = np.argmax(mineral_concentrations) if mineral_concentrations else -1
                
                self.get_logger().info(
                    f'💎 MINÉRAL! Type {mineral_type}, '
                    f'Conc={max_concentration:.3f}, Reward={reward:.1f}'
                )
                
                self.minerals_collected += 1
            
            return reward
            
        except Exception as e:
            self.get_logger().error(f'❌ Reward error: {e}')
            return 0.0
    
    def is_episode_done(self):
        """Vérifie si l'épisode est terminé.
        Basé uniquement sur le nombre de pas — l'objectif est d'explorer
        systématiquement, pas de s'arrêter après N minéraux trouvés.
        """
        return self.steps >= 300
    
    def publish_experience(self, action, reward, done):
        """Publie l'expérience pour le trainer central"""
        try:
            local_map, norm_pos = self.get_local_observation()
            
            experience = {
                'robot_id': self.robot_id,
                'step_data': {
                    'mineral_map': local_map.tolist(),
                    'position': norm_pos.tolist(),
                    'action': action,
                    'reward': reward,
                    'done': done,
                    'global_state': self.global_state.tolist() if self.global_state is not None else [],
                    'timestamp': time.time()
                }
            }
            
            msg = String()
            msg.data = json.dumps(experience)
            self.experience_pub.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f'❌ Error publishing experience: {e}')
    
    def publish_position(self):
        """Publie la position"""
        msg = Pose2D()
        msg.x = float(self.current_position[0])
        msg.y = float(self.current_position[1])
        msg.theta = 0.0
        self.position_pub.publish(msg)
    
    def publish_cleaning_action(self, x, y):
        """Publie l'action de nettoyage"""
        msg = Float32MultiArray()
        msg.data = [float(x), float(y)]
        self.cleaning_pub.publish(msg)
    
    def publish_discovery(self, x, y):
        """Publie une découverte"""
        msg = Float32MultiArray()
        mineral_data = self.mineral_map[y, x, :].tolist()
        msg.data = [float(self.robot_id), float(x), float(y)] + mineral_data
        self.discovery_pub.publish(msg)
    
    def publish_velocity(self, linear_x=0.1, angular_z=0.0):
        """Publie la vitesse"""
        msg = Twist()
        msg.linear.x = linear_x
        msg.angular.z = angular_z
        self.velocity_pub.publish(msg)
    
    def publish_status(self):
        """Publie le statut"""
        status_text = (f"Robot {self.robot_id} - Steps: {self.steps}, "
                       f"Reward: {self.episode_reward:.1f}, "
                       f"Minerals: {self.minerals_collected}, "
                       f"ε: {self.epsilon:.3f}")
        
        msg = String()
        msg.data = status_text
        self.status_pub.publish(msg)
        
        if self.steps % 20 == 0:
            self.get_logger().info(f'📈 {status_text}')

def parse_robot_id_from_argv(argv):
    """Parse le robot_id des arguments"""
    robot_id = 0
    if len(argv) > 1:
        arg = argv[1]
        try:
            robot_id = int(arg)
            return robot_id
        except ValueError:
            nums = re.findall(r'\d+', arg)
            if nums:
                return int(nums[0])
    
    env_id = os.getenv('ROBOT_ID')
    if env_id is not None:
        try:
            return int(env_id)
        except ValueError:
            pass
    return robot_id

def main(argv=None):
    rclpy.init(args=argv)
    argv = argv if argv is not None else sys.argv
    robot_id = parse_robot_id_from_argv(argv)
    
    node = None
    try:
        node = QMIXAgentNode(robot_id=robot_id)
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node:
            node.get_logger().info('🛑 Shutting down...')
    except Exception as e:
        if node:
            node.get_logger().error(f'❌ Fatal error: {e}')
        import traceback
        traceback.print_exc()
    finally:
        if node:
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()