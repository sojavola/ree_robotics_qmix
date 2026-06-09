#!/usr/bin/env python3
"""
Noeud agent QMIX décentralisé avec 2 contributions PhD :

  1. Multi-Scale (Observation multi-échelle)
     → Fenêtre locale 20×20 (détails fins)
     → Fenêtre régionale 60×60 sous-échantillonnée à 20×20 (contexte)

  2. GeoCommQMIX (Communication apprise)
     → Publie un message (features encodées) via ROS2
     → Reçoit les messages des N-1 autres agents
     → Le CommModule fusionne les messages avec gate + attention

  Coverage bonus (remplace Geo-ICM) :
     → r_coverage = coverage_weight par nouvelle cellule visitée dans l'épisode
     → Calculé dans science_reward_system.py — stable, pas bruité

Respect CTDE :
  - L'agent utilise UNIQUEMENT ses observations locales + messages reçus
  - L'état global n'est JAMAIS utilisé pour choisir une action
  - Le mixing network est dans le trainer (pas ici)
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
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
import threading
import queue
from collections import deque

from .networks import QMixLocalNetwork
from .config import QMIXConfig
from .science_reward_system import RealMineralRewardSystem


class QMIXAgentNode(Node):
    """Noeud agent QMIX décentralisé avec Multi-Scale + Communication + Coverage bonus."""

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

        # Système de récompenses (coverage_weight depuis config)
        self.reward_system = RealMineralRewardSystem(
            grid_size=(self.map_height, self.map_width),
            robot_id=robot_id,
            coverage_weight=self.config.coverage_weight
        )
        self.visited_positions = set()
        self.steps_without_mineral = 0

        # ── Fix A : callback groups AVANT _init_communication() qui les utilise ─
        # _cb_decision : callbacks qui lisent/écrivent l'état partagé (sérialisés)
        # _cb_fast     : callbacks rapides (status, weights) — peuvent tourner
        #                en parallèle de _cb_decision
        self._cb_decision = MutuallyExclusiveCallbackGroup()
        self._cb_fast     = MutuallyExclusiveCallbackGroup()

        # Réseau local QMIX (avec multi-scale + comm)
        self._init_local_network()

        # === CONTRIBUTION 2 : Communication ===
        self._init_communication()

        # Exploration
        self.epsilon = self.config.epsilon_start
        self.steps = 0
        self.episode_reward = 0
        self.total_reward = 0
        self.minerals_collected = 0

        # ── State machine (Fix A) ──────────────────────────────────────
        # ACTIVE       : décisions normales
        # DONE_WAITING : done=True envoyé, attente reset trainer (timeout 45s)
        # RESETTING    : reset en cours, transition vers ACTIVE
        self._agent_state = 'ACTIVE'
        self._done_time = None           # timestamp entrée DONE_WAITING
        self._state_entered_time = time.time()
        self._DONE_WAIT_TIMEOUT = 45.0   # secondes avant auto-reset si trainer muet

        # ── Stuck / collision detection (Fix D) ───────────────────────
        self._consecutive_collisions = 0
        self._steps_no_move = 0
        self._recovery_steps_remaining = 0
        self._COLLISION_THRESHOLD = 5    # collisions consécutives → recovery
        self._NO_MOVE_THRESHOLD = 8      # steps sans déplacement → recovery
        self._RECOVERY_DURATION = 15     # Fix B : durée recovery augmentée (10→15)

        # ── Non-blocking weight loader (Fix B) ────────────────────────
        # Queue de taille 1 : on ne garde que le path le plus récent
        self._weight_queue: queue.Queue = queue.Queue(maxsize=1)
        self._weight_lock = threading.Lock()
        # Fix I (O-10) — verrou pour discovered_map
        # Toutes les écritures/lectures sont dans _cb_decision (sérialisé),
        # mais le verrou rend l'invariant explicite et protège si _cb_fast
        # accède un jour à discovered_map (e.g. debug/status callback).
        self._map_lock = threading.Lock()
        self._weight_loader_thread = threading.Thread(
            target=self._weight_loader_worker, daemon=True
        )
        self._weight_loader_thread.start()

        # ── Observation cache (Fix A) ─────────────────────────────────
        # Evite de recalculer les observations deux fois par step
        # (une fois dans choose_action, une fois dans publish_experience)
        self._cached_local_obs  = None
        self._cached_regional_obs = None
        self._cached_norm_pos   = None

        # ── Fix B : direction tracking pour escape heuristic ─────────
        self._last_collision_direction = -1      # action qui a causé la dernière collision
        self._action_collision_counts  = np.zeros(8, dtype=np.int32)  # par action

        # ── Debug instrumentation (Fix G) ─────────────────────────────
        self._last_decision_time = time.time()
        self._last_heartbeat = time.time()
        self._heartbeat_warn_interval = 10.0  # warn si make_decision muet > 10s

        # ── Performance stats (Fix F) ─────────────────────────────────
        self._latency_history    = deque(maxlen=60)   # dernières 60 latences (s)
        self._stuck_count_ep     = 0                  # stuck events cet épisode
        self._steps_perf_window  = 0                  # steps depuis dernier log perf
        self._perf_window_start  = time.time()
        self._last_obs_ms        = 0.0                # durée dernière extraction obs (ms)
        self._last_fwd_ms        = 0.0                # durée dernier forward pass (ms)

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

        # Subscribers — callback groups pour MultiThreadedExecutor
        # _cb_decision : modifient/lisent l'état partagé → sérialisés
        # _cb_fast     : rapides, n'accèdent pas à l'état de décision
        self.mineral_sub = self.create_subscription(
            Float32MultiArray, self.config.mineral_map_topic,
            self.mineral_callback, 10,
            callback_group=self._cb_decision
        )
        self.obstacle_sub = self.create_subscription(
            OccupancyGrid, self.config.obstacle_map_topic,
            self.obstacle_callback, 10,
            callback_group=self._cb_decision
        )
        self.weight_sub = self.create_subscription(
            String, self.config.trainer_update_topic,
            self.weight_update_callback, 10,
            callback_group=self._cb_fast   # juste un enqueue, pas d'état partagé
        )
        self.epsilon_sub = self.create_subscription(
            Float32, self.config.trainer_epsilon_topic,
            self.epsilon_callback, 10,
            callback_group=self._cb_fast
        )
        self.episode_reset_sub = self.create_subscription(
            String, '/episode_reset',
            self._episode_reset_callback, 10,
            callback_group=self._cb_decision
        )

        # Timers
        self.decision_timer  = self.create_timer(
            0.2, self.make_decision, callback_group=self._cb_decision
        )
        self.status_timer    = self.create_timer(
            5.0, self.publish_status, callback_group=self._cb_fast
        )
        self.position_timer  = self.create_timer(
            0.2, self.publish_position, callback_group=self._cb_fast
        )
        self.debug_timer     = self.create_timer(
            30.0, self._debug_state_log, callback_group=self._cb_fast
        )

        self.get_logger().info(
            f'QMIX Agent {robot_id} initialized '
            f'[MultiScale={self.config.use_multi_scale}, '
            f'Comm={self.config.use_comm}, '
            f'CoverageW={self.config.coverage_weight}]'
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
                10,
                callback_group=self._cb_decision
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

        # Fix O-17 — modèle de capteur réaliste (XRF/spectromètre).
        # Avant : copie exacte de mineral_map (capteur parfait).
        # Après : ajout de bruit gaussien N(0, sensor_noise_std) clipé à [0,1].
        # sensor_noise_std=0.02 → ±2% d'erreur typique, cohérent avec
        # la précision d'un spectromètre XRF portable sur le terrain.
        noise_std = self.config.sensor_noise_std
        with self._map_lock:
            for iy in range(y_min, y_max):
                for ix in range(x_min, x_max):
                    dist = np.sqrt((ix - x) ** 2 + (iy - y) ** 2)
                    if dist <= r:
                        reading = self.mineral_map[iy, ix, :]
                        if noise_std > 0.0:
                            noise = np.random.normal(0.0, noise_std, reading.shape)
                            reading = np.clip(reading + noise, 0.0, 1.0)
                        self.discovered_map[iy, ix, :] = reading

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
        """Enqueue le path du checkpoint — chargement dans un thread dédié (Fix B).

        Le callback ROS2 ne fait AUCUN I/O : il enqueue le path et retourne
        immédiatement, libérant l'executor pour les autres callbacks/timers.
        """
        try:
            data = json.loads(msg.data)
            if data['type'] == 'weight_update':
                path = data['path']
                # Queue de taille 1 : si déjà pleine, on remplace par le plus récent
                try:
                    self._weight_queue.get_nowait()
                except queue.Empty:
                    pass
                self._weight_queue.put_nowait(path)
        except Exception as e:
            self.get_logger().error(f'Error enqueuing weight path: {e}')

    def _weight_loader_worker(self):
        """Thread dédié au chargement des poids (Fix B).

        Tourne en daemon — s'arrête proprement avec le process.
        Charge les poids depuis le fichier, protège le swap avec un Lock.
        """
        while True:
            try:
                path = self._weight_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            t0 = time.time()
            try:
                # Retry une fois si le fichier est partiellement écrit
                for attempt in range(2):
                    try:
                        checkpoint = torch.load(
                            path, map_location=self.device, weights_only=True
                        )
                        break
                    except Exception as load_err:
                        if attempt == 0:
                            time.sleep(0.1)  # attendre fin d'écriture
                        else:
                            raise load_err

                state_dict = checkpoint['state_dict']
                local_state_dict = {
                    key.replace('shared_agent_network.', ''): value
                    for key, value in state_dict.items()
                    if key.startswith('shared_agent_network.')
                }

                with self._weight_lock:
                    self.local_network.load_state_dict(local_state_dict)
                    self.local_network.eval()

                elapsed = time.time() - t0
                # Fix G : warn si chargement trop lent
                if elapsed > 0.5:
                    self.get_logger().warn(
                        f'[Robot {self.robot_id}] Weight load slow: {elapsed:.2f}s '
                        f'(step={checkpoint.get("train_step","?")})'
                    )
                else:
                    self.get_logger().debug(
                        f'Poids mis a jour en {elapsed*1000:.0f}ms '
                        f'(step={checkpoint.get("train_step","?")})'
                    )

            except Exception as e:
                self.get_logger().error(
                    f'[Robot {self.robot_id}] Weight load failed: {e}'
                )

    def epsilon_callback(self, msg: Float32):
        self.epsilon = float(msg.data)

    def _episode_reset_callback(self, msg: String):
        """Reçoit un reset d'épisode du trainer (watchdog OU fin normale).

        Fix C : suppression du filtre watchdog-only.
        Fix A : transition d'état selon l'état courant.
        """
        try:
            data = json.loads(msg.data)
            is_watchdog = data.get('watchdog', False)
            ep = data.get('episode', '?')

            if self._agent_state == 'DONE_WAITING':
                self.get_logger().info(
                    f'[Robot {self.robot_id}] Reset reçu (watchdog={is_watchdog}) '
                    f'— DONE_WAITING → RESETTING (ep={ep})'
                )
            elif self._agent_state == 'ACTIVE':
                self.get_logger().warn(
                    f'[Robot {self.robot_id}] Reset inattendu (watchdog={is_watchdog}) '
                    f'en état ACTIVE — reset forcé (ep={ep})'
                )
            # RESETTING → RESETTING : idempotent, on applique quand même

            self._do_reset()

        except Exception as e:
            self.get_logger().error(f'Error in episode_reset_callback: {e}')

    def _do_reset(self):
        """Effectue le reset complet de l'agent et transite vers ACTIVE (Fix A)."""
        self._agent_state = 'RESETTING'
        self._state_entered_time = time.time()

        self.reward_system.reset_episode()
        self.episode_reward = 0
        self.steps = 0
        self.minerals_collected = 0
        self.visited_positions = set()
        self.exploration_map = np.zeros(
            (self.map_height, self.map_width), dtype=np.float32
        )
        # Fix I (O-10) — reset de discovered_map sous verrou
        with self._map_lock:
            self.discovered_map = np.zeros(
                (self.map_height, self.map_width, 4), dtype=np.float32
            )
            # Fix 5a : NE PAS zéroïser mineral_map au reset.
            # Ancien comportement : mineral_map=zeros → make_decision bloquait en
            # "En attente de /mineral_map" jusqu'à la prochaine publication du serveur.
            # Si le watchdog (sync_timeout) se redéclenchait avant que le serveur publie,
            # un nouveau reset revenait à zeros → boucle infinie, 0 épisodes.
            # Nouveau comportement : on garde la carte précédente. Les 1-3 premiers
            # steps d'un nouvel épisode utilisent l'ancienne carte (légère race-condition
            # acceptable) mais le robot peut décider immédiatement après chaque reset.
        self.prev_features = None
        self.prev_action = None

        # Reset stuck counters
        self._consecutive_collisions = 0
        self._steps_no_move = 0
        self._recovery_steps_remaining = 0

        # Fix F : reset perf stats + obs cache entre épisodes
        self._stuck_count_ep = 0
        self._latency_history.clear()
        self._steps_perf_window = 0
        self._perf_window_start = time.time()
        self._cached_local_obs    = None
        self._cached_regional_obs = None
        self._cached_norm_pos     = None

        self._agent_state = 'ACTIVE'
        self._state_entered_time = time.time()
        self.get_logger().info(
            f'[Robot {self.robot_id}] État → ACTIVE'
        )

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
        # Fix I (O-10) — lecture de discovered_map sous verrou
        with self._map_lock:
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
        # Fix I (O-10) — lecture de discovered_map sous verrou
        with self._map_lock:
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
            # ── Fix A : construire les tenseurs HORS du lock ──────────
            # La construction des tenseurs ne touche pas au réseau — pas besoin
            # du lock. On réduit la durée de contention du ~50%.
            _t_obs = time.time()
            local_map, norm_pos = self.get_local_observation()
            regional_map        = self.get_regional_observation()

            # Cacher pour publish_experience (évite double calcul)
            self._cached_local_obs   = local_map
            self._cached_regional_obs = regional_map
            self._cached_norm_pos    = norm_pos

            map_tensor = torch.FloatTensor(
                local_map
            ).permute(2, 0, 1).unsqueeze(0).to(self.device)
            pos_tensor = torch.FloatTensor(norm_pos).unsqueeze(0).to(self.device)

            reg_tensor = None
            if regional_map is not None:
                reg_tensor = torch.FloatTensor(
                    regional_map
                ).permute(2, 0, 1).unsqueeze(0).to(self.device)

            comm_tensor = self._get_comm_messages_tensor()
            _t_net = time.time()

            # ── Fix A : lock minimal — uniquement le forward pass ─────
            with self._weight_lock:
                with torch.no_grad():
                    q_values = self.local_network(
                        map_tensor, pos_tensor,
                        regional_map=reg_tensor,
                        received_messages=comm_tensor
                    )
                    if self.config.use_comm:
                        features = self.local_network.encode(map_tensor, reg_tensor)

            _t_done = time.time()
            # Fix F : accumuler les temps pour perf stats
            self._last_obs_ms  = (_t_net  - _t_obs)  * 1000
            self._last_fwd_ms  = (_t_done - _t_net)  * 1000

            if self.config.use_comm:
                self._publish_comm_message(features)

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

    # ══════════════════════════════════════════════════════════════════
    #  BOUCLE DE DÉCISION
    # ══════════════════════════════════════════════════════════════════

    def make_decision(self):
        """Prend une décision de mouvement.

        Fix A : garde de state machine — bloque si DONE_WAITING, vérifie timeout.
        Fix G : mesure la latence de décision, heartbeat pour détection stall.
        """
        _step_start = time.time()

        # ── Fix G : heartbeat ──────────────────────────────────────────
        self._last_heartbeat = _step_start

        # ── Fix A : state machine guard ───────────────────────────────
        if self._agent_state == 'DONE_WAITING':
            elapsed = _step_start - self._done_time
            if elapsed >= self._DONE_WAIT_TIMEOUT:
                self.get_logger().warn(
                    f'[Robot {self.robot_id}] Timeout DONE_WAITING ({elapsed:.1f}s) '
                    f'— auto-reset sans trainer'
                )
                self._do_reset()
            else:
                return  # attente normale

        if self._agent_state == 'RESETTING':
            return  # reset en cours, ne pas décider

        # ── Guard carte vide ──────────────────────────────────────────
        if np.max(self.mineral_map) == 0:
            # Avertir toutes les 15s si le serveur ne publie pas
            if not hasattr(self, '_last_server_warn'):
                self._last_server_warn = 0.0
            if _step_start - self._last_server_warn > 15.0:
                self._last_server_warn = _step_start
                self.get_logger().warn(
                    f'[Robot {self.robot_id}] En attente de /mineral_map '
                    f'(serveur REE non connecté ?). '
                    f'Lancer: ros2 run ree_exploration_server server_node'
                )
            return

        # ── Décision ──────────────────────────────────────────────────
        action = self.choose_action()
        reward_ext, done = self.execute_action(action)

        # reward_ext inclut déjà le coverage bonus (science_reward_system)
        self.episode_reward += reward_ext
        self.total_reward += reward_ext
        self.steps += 1

        # Envoyer l'expérience au trainer (inchangé — CTDE intact)
        self.publish_experience(action, reward_ext, done)

        if done:
            # Fix A : transition ACTIVE → DONE_WAITING
            # On ne reset PAS ici — on attend le signal du trainer
            self._agent_state = 'DONE_WAITING'
            self._done_time = time.time()
            self._state_entered_time = self._done_time
            self.get_logger().info(
                f'[Robot {self.robot_id}] État → DONE_WAITING '
                f'(step={self.steps}, reward={self.episode_reward:.1f})'
            )

        if self.steps % 10 == 0:
            self.get_logger().debug(
                f'Step {self.steps}: Act={action}, '
                f'R={reward_ext:.2f}, eps={self.epsilon:.3f}'
            )

        # ── Fix F/G : latence décision ────────────────────────────────
        _elapsed = time.time() - _step_start
        self._latency_history.append(_elapsed)
        self._steps_perf_window += 1
        if _elapsed > 0.2:
            self.get_logger().warn(
                f'[Robot {self.robot_id}] Decision latency: {_elapsed*1000:.0f}ms '
                f'(obs={self._last_obs_ms:.0f}ms fwd={self._last_fwd_ms:.0f}ms '
                f'step={self.steps})'
            )

    def execute_action(self, action):
        """Exécute une action.

        Fix D : override de l'action par recovery si stuck détecté.
        Sticky counters : _consecutive_collisions, _steps_no_move.
        """
        x, y = self.current_position

        # ── Fix D/B : recovery override ───────────────────────────────
        if self._recovery_steps_remaining > 0:
            # Fix B : escape heuristic — préférer les directions cardinales
            # qui n'ont pas causé de collision récente
            safe_actions = [
                a for a in range(4)
                if self._action_collision_counts[a] == 0
            ]
            action = random.choice(safe_actions) if safe_actions else random.randint(0, 3)
            self._recovery_steps_remaining -= 1
            if self._recovery_steps_remaining == 0:
                # Reset compteurs collision pour repartir sans biais
                self._action_collision_counts[:] = 0
                self.get_logger().info(
                    f'[Robot {self.robot_id}] Recovery terminée — retour ACTIVE'
                )

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
            self._sensor_scan()
            reward = self.calculate_reward(new_x, new_y)
            done = self.is_episode_done()
            self.publish_cleaning_action(new_x, new_y)

            if reward > 50.0:
                self.publish_discovery(new_x, new_y)

            # ── Fix D : reset compteurs collision, vérif no-move ──────
            self._consecutive_collisions = 0
            if self.current_position == self.last_position:
                self._steps_no_move += 1
            else:
                self._steps_no_move = 0
        else:
            reward = self.config.penalty_collision
            done = self.is_episode_done()

            # ── Fix D/B : incrémenter compteur collision + tracking direction ─
            self._consecutive_collisions += 1
            self._steps_no_move += 1
            self._last_collision_direction = action
            self._action_collision_counts[action % len(direction_vectors)] += 1

        # ── Fix D : déclencher recovery si seuils atteints ───────────
        if (self._recovery_steps_remaining == 0 and
                (self._consecutive_collisions >= self._COLLISION_THRESHOLD or
                 self._steps_no_move >= self._NO_MOVE_THRESHOLD)):
            self._recovery_steps_remaining = self._RECOVERY_DURATION
            self._consecutive_collisions = 0
            self._steps_no_move = 0
            self._stuck_count_ep += 1  # Fix F : compter les events stuck
            self.get_logger().warn(
                f'[Robot {self.robot_id}] Stuck détecté — recovery {self._RECOVERY_DURATION} steps '
                f'(stuck_ep={self._stuck_count_ep})'
            )

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
        """Encode un numpy array float32 en base64 (3× plus compact que .tolist()).

        Fix A : évite la copie .astype() si l'array est déjà float32.
        """
        buf = arr.tobytes() if arr.dtype == np.float32 else arr.astype(np.float32).tobytes()
        return base64.b64encode(buf).decode('ascii')

    def publish_experience(self, action, reward_ext, done):
        """Publie l'expérience pour le trainer central (arrays encodés en base64).

        Fix A : utilise les observations cachées par choose_action() au lieu de
        les recalculer — économise ~30ms par step.
        reward_ext inclut déjà le coverage bonus calculé dans science_reward_system.
        """
        try:
            if self._cached_local_obs is not None:
                local_map    = self._cached_local_obs
                norm_pos     = self._cached_norm_pos
                regional_map = self._cached_regional_obs
            else:
                local_map, norm_pos = self.get_local_observation()
                regional_map = self.get_regional_observation()

            experience = {
                'robot_id': self.robot_id,
                'step_data': {
                    'mineral_map': self._arr_to_b64(local_map),
                    'position': norm_pos.tolist(),
                    'action': int(action),
                    'reward': float(reward_ext),
                    'done': bool(done),
                    # global_state supprimé des messages — le trainer le construit
                    # lui-même en souscrivant à /mineral_map + /obstacle_map.
                    # Cela réduit la taille des messages de ~350KB à ~25KB,
                    # évitant les pertes de paquets DDS sous charge CPU.
                    'timestamp': time.time()
                }
            }

            # Priorité 4 — envoyer les composantes du reward avec le dernier step
            # pour le logging TensorBoard (Reward/Mineral, Reward/Coverage, Reward/Penalty)
            if done:
                experience['step_data']['reward_components'] = (
                    self.reward_system.get_episode_reward_components()
                )

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

    def _debug_state_log(self):
        """Fix G/F : log d'état toutes les 30s, indépendant de steps.

        Permet de voir l'état même si make_decision est bloqué/en attente.
        Détecte aussi un stall de l'executor (heartbeat absent).
        Affiche les métriques de performance de la fenêtre courante (Fix F).
        """
        now = time.time()
        time_in_state = now - self._state_entered_time

        self.get_logger().info(
            f'[Robot {self.robot_id}] STATE={self._agent_state} '
            f'depuis {time_in_state:.1f}s | '
            f'steps={self.steps} eps={self.epsilon:.3f} '
            f'collisions={self._consecutive_collisions} '
            f'no_move={self._steps_no_move} '
            f'recovery={self._recovery_steps_remaining}'
        )

        # Fix F : performance stats sur la fenêtre de 30s
        if self._latency_history:
            lat = np.array(self._latency_history)
            avg_lat = float(np.mean(lat)) * 1000
            max_lat = float(np.max(lat)) * 1000
        else:
            avg_lat = max_lat = 0.0

        window_elapsed = now - self._perf_window_start
        steps_per_sec = self._steps_perf_window / max(window_elapsed, 1.0)

        self.get_logger().info(
            f'[Robot {self.robot_id}] PERF | '
            f'latence avg={avg_lat:.0f}ms max={max_lat:.0f}ms | '
            f'steps/s={steps_per_sec:.2f} | '
            f'stuck_ep={self._stuck_count_ep} | '
            f'obs={self._last_obs_ms:.0f}ms fwd={self._last_fwd_ms:.0f}ms'
        )
        # Réinitialiser la fenêtre de mesure
        self._steps_perf_window = 0
        self._perf_window_start = now

        # Fix G : stall detector — heartbeat non mis à jour = executor bloqué
        heartbeat_age = now - self._last_heartbeat
        if heartbeat_age > self._heartbeat_warn_interval and self._agent_state == 'ACTIVE':
            self.get_logger().warn(
                f'[Robot {self.robot_id}] EXECUTOR STALL détecté : '
                f'make_decision muet depuis {heartbeat_age:.1f}s en état ACTIVE'
            )


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
    executor = None
    try:
        node = QMIXAgentNode(robot_id=robot_id)
        # Fix A : MultiThreadedExecutor — _cb_decision sérialisé (état partagé),
        # _cb_fast en parallèle (status, weights, position).
        # 4 threads : 2 groupes × 2 pour absorber les pics de callbacks.
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        if node:
            node.get_logger().info('Shutting down...')
    except Exception as e:
        if node:
            node.get_logger().error(f'Fatal error: {e}')
        import traceback
        traceback.print_exc()
    finally:
        if executor:
            executor.shutdown()
        if node:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
