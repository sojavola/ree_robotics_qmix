#!/usr/bin/env python3
"""
Noeud agent QMIX décentralisé avec 3 contributions PhD :

  1. Geo-ICM (Curiosité intrinsèque coopérative)
     → Forward model prédit features t+1 ; erreur = bonus de curiosité
     → r_total = r_extrinsic + β * r_curiosity

  2. Multi-Scale (Observation multi-échelle)
     → Fenêtre locale 20×20 (détails fins)
     → Fenêtre régionale 60×60 sous-échantillonnée à 20×20 (contexte)

  3. GeoCommQMIX (Communication apprise)
     → Publie un message (features encodées) via ROS2
     → Reçoit les messages des N-1 autres agents
     → Le CommModule fusionne les messages avec gate + attention

Respect CTDE :
  - L'agent utilise UNIQUEMENT ses observations locales + messages reçus
  - L'état global n'est JAMAIS utilisé pour choisir une action
  - Le mixing network est dans le trainer (pas ici)
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String, Int32, Float32
from geometry_msgs.msg import Pose2D, Twist
from nav_msgs.msg import OccupancyGrid
import numpy as np
import torch
import json
import base64
import random
import os
import sys
import re
import time

from .networks import QMixLocalNetwork
from .config import QMIXConfig
from .science_reward_system import RealMineralRewardSystem
from .geo_icm import GeoICM


class QMIXAgentNode(Node):
    """Noeud agent QMIX décentralisé avec ICM + Multi-Scale + Communication."""

    def __init__(self, robot_id=0):
        super().__init__(f'qmix_agent_{robot_id}')

        self.robot_id = robot_id
        self.get_logger().info(f'Robot ID: {robot_id}')

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
        # mineral_map = carte COMPLÈTE reçue du serveur (utilisée pour reward + global_state)
        self.mineral_map     = np.zeros((self.map_height, self.map_width, 4), dtype=np.float32)
        # discovered_map = ce que le robot a RÉELLEMENT découvert avec ses capteurs
        # Initialement VIDE — remplie progressivement par le capteur (sensor_range)
        self.discovered_map  = np.zeros((self.map_height, self.map_width, 4), dtype=np.float32)
        self.obstacle_map    = np.zeros((self.map_height, self.map_width), dtype=np.float32)
        self.exploration_map = np.zeros((self.map_height, self.map_width), dtype=np.float32)
        self.global_state    = None
        self.sensor_range    = self.config.sensor_range

        # Position
        self.current_position = self.get_initial_position()
        self.last_position = self.current_position

        # Système de récompenses
        self.reward_system = RealMineralRewardSystem(
            grid_size=(self.map_height, self.map_width),
            robot_id=robot_id
        )
        self.visited_positions = set()
        self.steps_without_mineral = 0

        # Réseau local QMIX (avec multi-scale + comm)
        self._init_local_network()

        # === CONTRIBUTION 1 : Geo-ICM ===
        self._init_icm()

        # === CONTRIBUTION 3 : Communication ===
        self._init_communication()

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
        self.decision_timer = self.create_timer(0.5, self.make_decision)
        self.status_timer = self.create_timer(5.0, self.publish_status)
        self.position_timer = self.create_timer(0.5, self.publish_position)

        self.get_logger().info(
            f'QMIX Agent {robot_id} initialized '
            f'[ICM={self.config.use_icm}, '
            f'MultiScale={self.config.use_multi_scale}, '
            f'Comm={self.config.use_comm}]'
        )

    # ══════════════════════════════════════════════════════════════════
    #  INITIALISATION
    # ══════════════════════════════════════════════════════════════════

    def _init_local_network(self):
        """Initialise le réseau local (avec multi-scale + comm)."""
        self.local_network = QMixLocalNetwork(
            input_shape=self.config.state_shape,
            num_actions=self.num_actions,
            hidden_dim=self.config.hidden_dim,
            local_obs_size=self.config.local_obs_size,
            use_multi_scale=self.config.use_multi_scale,
            use_comm=self.config.use_comm,
            comm_dim=self.config.comm_dim,
            num_agents=self.config.num_robots
        ).to(self.device)
        self.local_network.eval()
        self.get_logger().info('Reseau local QMIX initialise (multi-scale + comm)')

    def _init_icm(self):
        """Initialise le module Geo-ICM (curiosité intrinsèque)."""
        if not self.config.use_icm:
            self.icm = None
            self.prev_features = None
            return

        self.icm = GeoICM(
            feature_dim=self.config.icm_feature_dim,
            action_dim=self.num_actions,
            hidden_dim=self.config.icm_hidden_dim,
            curiosity_weight=self.config.curiosity_weight
        ).to(self.device)
        self.icm.eval()

        # Features CNN du pas précédent (pour le forward model)
        self.prev_features = None
        self.prev_action = None

        self.get_logger().info(
            f'Geo-ICM initialise (beta={self.config.curiosity_weight})'
        )

    def _init_communication(self):
        """Initialise la communication inter-agents (GeoCommQMIX)."""
        if not self.config.use_comm:
            self.received_messages = {}
            return

        # Buffer des messages reçus des autres agents
        # {robot_id: np.array(comm_dim)}
        self.received_messages = {}

        # Publisher pour envoyer notre message
        self.comm_pub = self.create_publisher(
            Float32MultiArray,
            f'{self.config.comm_topic}/robot_{self.robot_id}',
            10
        )

        # Subscribers pour recevoir les messages des autres agents
        self.comm_subs = []
        for other_id in range(self.config.num_robots):
            if other_id == self.robot_id:
                continue
            sub = self.create_subscription(
                Float32MultiArray,
                f'{self.config.comm_topic}/robot_{other_id}',
                lambda msg, rid=other_id: self.comm_callback(msg, rid),
                10
            )
            self.comm_subs.append(sub)

        self.get_logger().info(
            f'Communication initialisee (dim={self.config.comm_dim}, '
            f'{self.config.num_robots - 1} peers)'
        )

    # ══════════════════════════════════════════════════════════════════
    #  CALLBACKS
    # ══════════════════════════════════════════════════════════════════

    def _build_global_state(self):
        """Construit l'état global (6, H, W) avec la vraie carte (pour CTDE)."""
        mineral_ch  = self.mineral_map.transpose(2, 0, 1)
        obstacle_ch = (self.obstacle_map > 0).astype(np.float32)[np.newaxis]
        explore_ch  = self.exploration_map[np.newaxis]
        self.global_state = np.concatenate(
            [mineral_ch, obstacle_ch, explore_ch], axis=0
        )

    def _sensor_scan(self):
        """
        Simule un capteur géologique (XRF/spectromètre).

        Le robot ne détecte les minéraux que dans un rayon = sensor_range
        autour de sa position actuelle. Les lectures sont ajoutées à
        la discovered_map (mémoire cumulative).
        """
        x, y = self.current_position
        r = self.sensor_range

        x_min = max(0, x - r)
        x_max = min(self.map_width, x + r + 1)
        y_min = max(0, y - r)
        y_max = min(self.map_height, y + r + 1)

        for iy in range(y_min, y_max):
            for ix in range(x_min, x_max):
                dist = np.sqrt((ix - x) ** 2 + (iy - y) ** 2)
                if dist <= r:
                    # Copier la vraie concentration dans la discovered_map
                    self.discovered_map[iy, ix, :] = self.mineral_map[iy, ix, :]

    def mineral_callback(self, msg: Float32MultiArray):
        data = np.array(msg.data, dtype=np.float32)
        if data.size == self.map_height * self.map_width * 4:
            self.mineral_map = data.reshape(self.map_height, self.map_width, 4)
            self._build_global_state()

    def obstacle_callback(self, msg: OccupancyGrid):
        data = np.array(msg.data, dtype=np.float32)
        if data.size == self.map_height * self.map_width:
            self.obstacle_map = data.reshape(self.map_height, self.map_width)
            self._build_global_state()

    def weight_update_callback(self, msg: String):
        """Met à jour les poids du réseau local + ICM."""
        try:
            data = json.loads(msg.data)
            if data['type'] == 'weight_update':
                path = data['path']
                checkpoint = torch.load(path, map_location=self.device, weights_only=True)

                # Réseau partagé → extraire 'shared_agent_network.*'
                state_dict = checkpoint['state_dict']
                local_state_dict = {
                    key.replace('shared_agent_network.', ''): value
                    for key, value in state_dict.items()
                    if key.startswith('shared_agent_network.')
                }
                self.local_network.load_state_dict(local_state_dict)
                self.local_network.eval()

                # ICM : charger les poids du forward model
                if self.config.use_icm and 'icm_state_dict' in checkpoint:
                    self.icm.load_state_dict(checkpoint['icm_state_dict'])
                    self.icm.eval()

                self.get_logger().debug(
                    f'Poids mis a jour (step {data["train_step"]})'
                )

        except Exception as e:
            self.get_logger().error(f'Error updating weights: {e}')

    def epsilon_callback(self, msg: Float32):
        self.epsilon = float(msg.data)

    def comm_callback(self, msg: Float32MultiArray, sender_id: int):
        """Reçoit un message de communication d'un autre agent."""
        self.received_messages[sender_id] = np.array(msg.data, dtype=np.float32)

    # ══════════════════════════════════════════════════════════════════
    #  OBSERVATIONS (multi-scale)
    # ══════════════════════════════════════════════════════════════════

    def get_initial_position(self):
        return (random.randint(5, self.map_width - 6),
                random.randint(5, self.map_height - 6))

    def get_local_observation(self):
        """
        Observation locale (20×20, 6 canaux) basée sur la DISCOVERED_MAP.

        Le CNN voit uniquement ce que le robot a découvert via ses capteurs,
        PAS la carte complète. Les zones non explorées = 0.
        """
        x, y = self.current_position
        window_size = self.config.local_obs_size
        half = window_size // 2

        x_min = max(0, x - half)
        x_max = min(self.map_width, x + half)
        y_min = max(0, y - half)
        y_max = min(self.map_height, y + half)

        h = y_max - y_min
        w = x_max - x_min

        local_map = np.zeros((window_size, window_size, 6), dtype=np.float32)
        # Canaux 0-3 : discovered_map (PAS mineral_map)
        local_map[:h, :w, 0:4] = self.discovered_map[y_min:y_max, x_min:x_max]
        local_map[:h, :w, 4]   = (self.obstacle_map[y_min:y_max, x_min:x_max] > 0).astype(np.float32)
        local_map[:h, :w, 5]   = self.exploration_map[y_min:y_max, x_min:x_max]

        norm_pos = np.array([x / self.map_width, y / self.map_height], dtype=np.float32)
        return local_map, norm_pos

    def get_regional_observation(self):
        """
        Observation régionale (60×60 sous-échantillonnée à 20×20).

        Contribution 2 : Multi-Scale.
        Couvre une zone 3× plus large que l'observation locale,
        sous-échantillonnée pour garder la même taille d'entrée CNN.
        """
        if not self.config.use_multi_scale:
            return None

        x, y = self.current_position
        regional_size = self.config.regional_obs_size  # 60
        local_size = self.config.local_obs_size        # 20
        half = regional_size // 2

        x_min = max(0, x - half)
        x_max = min(self.map_width, x + half)
        y_min = max(0, y - half)
        y_max = min(self.map_height, y + half)

        h = y_max - y_min
        w = x_max - x_min

        # Extraire la zone régionale brute (discovered_map, PAS mineral_map)
        regional_raw = np.zeros((regional_size, regional_size, 6), dtype=np.float32)
        regional_raw[:h, :w, 0:4] = self.discovered_map[y_min:y_max, x_min:x_max]
        regional_raw[:h, :w, 4]   = (self.obstacle_map[y_min:y_max, x_min:x_max] > 0).astype(np.float32)
        regional_raw[:h, :w, 5]   = self.exploration_map[y_min:y_max, x_min:x_max]

        # Sous-échantillonner à local_size × local_size (average pooling vectorisé)
        # reshape (60,60,6) → (20,3,20,3,6) puis mean sur axes 1 et 3
        scale = self.config.regional_scale  # 3
        regional_ds = (
            regional_raw[:local_size * scale, :local_size * scale, :]
            .reshape(local_size, scale, local_size, scale, 6)
            .mean(axis=(1, 3))
        ).astype(np.float32)

        return regional_ds

    def _get_comm_messages_tensor(self):
        """
        Construit le tenseur des messages reçus pour le CommModule.

        Returns:
            (1, N-1, comm_dim) ou None si pas de communication
        """
        if not self.config.use_comm:
            return None

        comm_dim = self.config.comm_dim
        num_others = self.config.num_robots - 1

        messages = []
        for other_id in range(self.config.num_robots):
            if other_id == self.robot_id:
                continue
            if other_id in self.received_messages:
                msg = self.received_messages[other_id]
                if len(msg) == comm_dim:
                    messages.append(torch.FloatTensor(msg))
                else:
                    messages.append(torch.zeros(comm_dim))
            else:
                messages.append(torch.zeros(comm_dim))

        if not messages:
            return None

        return torch.stack(messages).unsqueeze(0).to(self.device)  # (1, N-1, comm_dim)

    # ══════════════════════════════════════════════════════════════════
    #  CHOIX D'ACTION + CURIOSITÉ + COMMUNICATION
    # ══════════════════════════════════════════════════════════════════

    def choose_action(self):
        """Choisit une action selon epsilon-greedy avec multi-scale + comm."""
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)

        try:
            with torch.no_grad():
                local_map, norm_pos = self.get_local_observation()
                regional_map = self.get_regional_observation()

                # Tenseurs
                map_tensor = torch.FloatTensor(
                    local_map
                ).permute(2, 0, 1).unsqueeze(0).to(self.device)
                pos_tensor = torch.FloatTensor(norm_pos).unsqueeze(0).to(self.device)

                # Multi-scale
                reg_tensor = None
                if regional_map is not None:
                    reg_tensor = torch.FloatTensor(
                        regional_map
                    ).permute(2, 0, 1).unsqueeze(0).to(self.device)

                # Communication messages
                comm_tensor = self._get_comm_messages_tensor()

                # Q-values
                q_values = self.local_network(
                    map_tensor, pos_tensor,
                    regional_map=reg_tensor,
                    received_messages=comm_tensor
                )

                # Publier notre message de communication
                if self.config.use_comm:
                    features = self.local_network.encode(map_tensor, reg_tensor)
                    self._publish_comm_message(features)

                    # Stocker les features pour ICM
                    if self.config.use_icm:
                        self._current_features = features.squeeze(0)

                return q_values.squeeze().argmax().item()

        except Exception as e:
            self.get_logger().error(f'Action selection error: {e}')
            return random.randint(0, self.num_actions - 1)

    def _publish_comm_message(self, features):
        """Publie un message de communication encodé via ROS2."""
        if not self.config.use_comm:
            return

        try:
            with torch.no_grad():
                message = self.local_network.comm_module.encode_message(features)
                msg = Float32MultiArray()
                msg.data = message.squeeze(0).cpu().numpy().tolist()
                self.comm_pub.publish(msg)
        except Exception as e:
            self.get_logger().debug(f'Comm publish error: {e}')

    def _compute_curiosity_reward(self, action):
        """
        Calcule la récompense de curiosité (Contribution 1 : Geo-ICM).

        r_curiosity = β * ||φ̂(s_{t+1}) - φ(s_{t+1})||²

        Utilise les features CNN du pas précédent et du pas actuel.
        """
        if not self.config.use_icm or self.icm is None:
            return 0.0

        try:
            with torch.no_grad():
                local_map, _ = self.get_local_observation()
                regional_map = self.get_regional_observation()

                map_tensor = torch.FloatTensor(
                    local_map
                ).permute(2, 0, 1).unsqueeze(0).to(self.device)
                reg_tensor = None
                if regional_map is not None:
                    reg_tensor = torch.FloatTensor(
                        regional_map
                    ).permute(2, 0, 1).unsqueeze(0).to(self.device)

                # Features actuelles
                current_features = self.local_network.encode(map_tensor, reg_tensor)
                current_features = current_features.squeeze(0)

                if self.prev_features is not None and self.prev_action is not None:
                    action_tensor = torch.tensor(
                        self.prev_action, device=self.device
                    )
                    curiosity_reward = self.icm.compute_curiosity_reward(
                        self.prev_features, action_tensor, current_features
                    )
                    curiosity = float(curiosity_reward.item())
                else:
                    curiosity = 0.0

                # Mémoriser pour le prochain pas
                self.prev_features = current_features
                self.prev_action = action

                return curiosity

        except Exception as e:
            self.get_logger().debug(f'ICM error: {e}')
            return 0.0

    # ══════════════════════════════════════════════════════════════════
    #  BOUCLE DE DÉCISION
    # ══════════════════════════════════════════════════════════════════

    def make_decision(self):
        """Prend une décision de mouvement."""
        if np.max(self.mineral_map) == 0:
            return

        action = self.choose_action()
        reward_ext, done = self.execute_action(action)

        # Curiosité intrinsèque (Contribution 1)
        reward_curiosity = self._compute_curiosity_reward(action)
        reward_total = reward_ext + reward_curiosity

        self.episode_reward += reward_total
        self.total_reward += reward_total
        self.steps += 1

        # Envoyer l'expérience au trainer
        self.publish_experience(action, reward_ext, reward_curiosity, done)

        if done:
            self.reward_system.reset_episode()
            self.episode_reward = 0
            self.steps = 0
            self.minerals_collected = 0
            self.visited_positions = set()
            self.exploration_map = np.zeros(
                (self.map_height, self.map_width), dtype=np.float32
            )
            # Reset la carte découverte — nouvel épisode = nouvelle carte
            self.discovered_map = np.zeros(
                (self.map_height, self.map_width, 4), dtype=np.float32
            )
            self.prev_features = None
            self.prev_action = None

        if self.steps % 10 == 0:
            self.get_logger().debug(
                f'Step {self.steps}: Act={action}, '
                f'R_ext={reward_ext:.2f}, R_cur={reward_curiosity:.3f}, '
                f'eps={self.epsilon:.3f}'
            )

    def execute_action(self, action):
        """Exécute une action."""
        x, y = self.current_position

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
            self.exploration_map[new_y, new_x] = 1.0
            # Scanner les minéraux autour de la nouvelle position
            self._sensor_scan()
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
        return (0 <= x < self.map_width and
                0 <= y < self.map_height and
                int(self.obstacle_map[y, x]) == 0)

    def calculate_reward(self, x, y):
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

            max_concentration = max(mineral_concentrations) if mineral_concentrations else 0.0
            if max_concentration > 0.3:
                mineral_type = int(np.argmax(mineral_concentrations))
                self.get_logger().info(
                    f'MINERAL! Type {mineral_type}, '
                    f'Conc={max_concentration:.3f}, Reward={reward:.1f}'
                )
                self.minerals_collected += 1

            return reward

        except Exception as e:
            self.get_logger().error(f'Reward error: {e}')
            return 0.0

    def is_episode_done(self):
        return self.steps >= 300

    # ══════════════════════════════════════════════════════════════════
    #  PUBLICATION
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _arr_to_b64(arr):
        """Encode un numpy array float32 en base64 (3× plus compact que .tolist())."""
        return base64.b64encode(
            arr.astype(np.float32).tobytes()
        ).decode('ascii')

    def publish_experience(self, action, reward_ext, reward_curiosity, done):
        """Publie l'expérience pour le trainer central (arrays encodés en base64)."""
        try:
            local_map, norm_pos = self.get_local_observation()
            regional_map = self.get_regional_observation()

            experience = {
                'robot_id': self.robot_id,
                'step_data': {
                    'mineral_map': self._arr_to_b64(local_map),
                    'position': norm_pos.tolist(),
                    'action': int(action),
                    'reward': float(reward_ext),
                    'reward_curiosity': float(reward_curiosity),
                    'done': bool(done),
                    'global_state': (
                        self._arr_to_b64(self.global_state)
                        if self.global_state is not None else ''
                    ),
                    'timestamp': time.time()
                }
            }

            # Multi-scale : ajouter l'observation régionale
            if regional_map is not None:
                experience['step_data']['regional_map'] = self._arr_to_b64(regional_map)

            msg = String()
            msg.data = json.dumps(experience)
            self.experience_pub.publish(msg)

        except Exception as e:
            self.get_logger().error(f'Error publishing experience: {e}')

    def publish_position(self):
        msg = Pose2D()
        msg.x = float(self.current_position[0])
        msg.y = float(self.current_position[1])
        msg.theta = 0.0
        self.position_pub.publish(msg)

    def publish_cleaning_action(self, x, y):
        msg = Float32MultiArray()
        msg.data = [float(x), float(y)]
        self.cleaning_pub.publish(msg)

    def publish_discovery(self, x, y):
        msg = Float32MultiArray()
        mineral_data = self.mineral_map[y, x, :].tolist()
        msg.data = [float(self.robot_id), float(x), float(y)] + mineral_data
        self.discovery_pub.publish(msg)

    def publish_velocity(self, linear_x=0.1, angular_z=0.0):
        msg = Twist()
        msg.linear.x = linear_x
        msg.angular.z = angular_z
        self.velocity_pub.publish(msg)

    def publish_status(self):
        status_text = (
            f"Robot {self.robot_id} - Steps: {self.steps}, "
            f"Reward: {self.episode_reward:.1f}, "
            f"Minerals: {self.minerals_collected}, "
            f"eps: {self.epsilon:.3f}"
        )
        msg = String()
        msg.data = status_text
        self.status_pub.publish(msg)

        if self.steps % 20 == 0:
            self.get_logger().info(status_text)


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

def parse_robot_id_from_argv(argv):
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
            node.get_logger().info('Shutting down...')
    except Exception as e:
        if node:
            node.get_logger().error(f'Fatal error: {e}')
        import traceback
        traceback.print_exc()
    finally:
        if node:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
