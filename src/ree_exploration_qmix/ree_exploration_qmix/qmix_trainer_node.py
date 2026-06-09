#!/usr/bin/env python3
"""
Noeud d'entraînement centralisé QMIX avec 2 contributions PhD :

  1. Multi-Scale : Le forward pass utilise regional_maps si disponibles
     → Le CNN multi-scale est entraîné end-to-end via la loss QMIX

  2. GeoCommQMIX : Communication dans le forward pass centralisé
     → Pendant l'entraînement, les messages sont recalculés à chaque pas
     → Le CommModule (gate, attention) est entraîné end-to-end via QMIX

  Coverage bonus (contribution 3 remplacée) :
     → Remplace le Geo-ICM incompatible avec les cartes régénérées aléatoirement
     → r_coverage calculé dans science_reward_system.py (côté agent)
     → Stocké dans rewards — aucun réseau ni optimiseur supplémentaire nécessaire

Respect CTDE :
  - L'état global est utilisé UNIQUEMENT dans le mixing network
  - Les Q-values individuelles dépendent des observations locales + messages
  - La monotonie QMIX (IGM) est préservée (hypernetwork inchangé)
"""

import signal
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, ReliabilityPolicy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
import threading
from std_msgs.msg import Float32MultiArray, String, Int32, Float32
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import OccupancyGrid
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import os
import json
import base64
import time
import csv
import traceback
from collections import deque
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from .networks import QMixNetwork, QMixTargetNetwork
from .replay_buffer import QMIXReplayBuffer
from .config import QMIXConfig


class QMIXTrainerNode(Node):
    """Noeud d'entraînement centralisé QMIX avec Multi-Scale + Comm + Coverage bonus."""

    def __init__(self):
        super().__init__('qmix_trainer')

        # Charger la configuration
        self.declare_parameter('config_file', '')
        config_file = self.get_parameter('config_file').get_parameter_value().string_value

        if config_file:
            self.config = QMIXConfig.from_yaml(config_file)
        else:
            self.config = QMIXConfig()

        self.get_logger().info('Initializing QMIX Trainer Node')
        self.get_logger().info(f'Config: {self.config}')

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'Using device: {self.device}')

        # Initialiser les réseaux
        self._init_networks()

        # Replay buffer
        self.replay_buffer = QMIXReplayBuffer(
            buffer_size=self.config.buffer_size,
            batch_size=self.config.batch_size,
            num_robots=self.config.num_robots,
            device=self.device
        )

        # Buffer temporaire pour l'épisode en cours
        self.current_episode = self._init_episode_buffer()

        # Suivi done par robot — l'épisode se termine quand TOUS les robots
        # ont envoyé done=True (indépendamment du sync par pas)
        self._robot_done = {i: False for i in range(self.config.num_robots)}
        # Temps de début d'épisode — pour rejeter les done=True résiduels
        # par critère temporel (plus fiable que le comptage de steps)
        self._episode_start_time = time.time()
        self._last_episode_complete_time = time.time()  # watchdog épisode (Step 4)

        # Compteurs
        self.train_step = 0
        self.episode_count = 0
        self.global_step = 0
        self.epsilon = self.config.epsilon_start
        # Épisodes complétés dans la session courante (repart à 0 à chaque restart).
        # Utilisé pour le trigger de sauvegarde : on attend au moins 5 épisodes
        # de la session avant de sauvegarder (évite le save prématuré au 1er épisode
        # quand episode_count % 5 == 0 par coïncidence sur le compteur chargé du CSV).
        self._session_episodes = 0

        # Fenêtres glissantes MA-50 (désactivé — utiliser le slider Smoothing de TensorBoard)
        self._reward_window    = deque(maxlen=50)
        # self._minerals_window  = deque(maxlen=50)
        # self._mineral_r_window = deque(maxlen=50)
        # self._coverage_window  = deque(maxlen=50)
        # self._penalty_window   = deque(maxlen=50)
        # self._loss_window      = deque(maxlen=50)
        # self._gradnorm_window  = deque(maxlen=50)
        # self._qtot_window      = deque(maxlen=50)
        # self._eval_window      = deque(maxlen=50)
        # self._eval_rps_window  = deque(maxlen=50)
        # self._robot_r_windows  = [deque(maxlen=50) for _ in range(4)]
        # self._robot_m_windows  = [deque(maxlen=50) for _ in range(4)]

        # === EVAL SPLIT ===
        self.eval_freq = 20
        self.eval_episodes = 1
        self.is_eval_mode = False
        self.eval_episode_count = 0
        self.eval_round = 0
        self.eval_rewards_accum = []
        self.eval_steps_accum = []

        # Fix G (Change 2) — callback groups + lock pour MultiThreadedExecutor
        # Group A : experience (tourne en parallèle du training)
        # Group B : training + timers lents (mutuellement exclusifs entre eux)
        self._cb_experience = MutuallyExclusiveCallbackGroup()
        self._cb_train = MutuallyExclusiveCallbackGroup()
        # Protège replay_buffer contre accès concurrent add/sample
        self._replay_lock = threading.Lock()
        # Fix I (O-2) — verrou non-réentrant pour _end_episode()
        # acquire(blocking=False) → rejet immédiat si déjà tenu (defense-in-depth).
        # Avec _cb_experience sérialisé ce cas n'arrive pas en pratique, mais le
        # verrou rend l'invariant explicite et prévient toute régression future.
        self._end_episode_mutex = threading.Lock()
        # Fix I (O-10) — verrou pour _trainer_global_state
        # Écrit par _trainer_mineral/obstacle_callback (Group A),
        # lu par experience_callback (Group A) → déjà sérialisé dans le groupe.
        # Le verrou protège si train_step_callback (Group B) accède un jour à
        # _trainer_global_state directement (évolution future).
        self._map_lock = threading.Lock()

        # Publishers
        self.episode_reset_pub = self.create_publisher(
            String, '/episode_reset', 10
        )
        self.weight_update_pub = self.create_publisher(
            String, self.config.trainer_update_topic, 10
        )
        self.epsilon_pub = self.create_publisher(
            Float32, self.config.trainer_epsilon_topic, 10
        )

        # Subscribers — grande profondeur pour ne pas dropper les messages
        # pendant les blocs d'entraînement CPU (~30s/step).
        # 4 robots × 300 steps/épisode = 1200 messages max en attente.
        _exp_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1500,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.exp_sub = self.create_subscription(
            String, self.config.agent_exp_topic,
            self.experience_callback, _exp_qos,
            callback_group=self._cb_experience
        )

        # Le trainer maintient son propre global_state en souscrivant directement
        # aux cartes du serveur. Les agents n'envoient plus le global_state dans
        # leurs messages d'expérience (trop grand : ~350KB par message).
        self._trainer_global_state = np.zeros(
            self.config.state_shape, dtype=np.float32
        )
        self._trainer_mineral_map = np.zeros(
            (self.config.map_height, self.config.map_width, 4), dtype=np.float32
        )
        self._trainer_obstacle_map = np.zeros(
            (self.config.map_height, self.config.map_width), dtype=np.float32
        )
        self.create_subscription(
            Float32MultiArray, self.config.mineral_map_topic,
            self._trainer_mineral_callback, 10,
            callback_group=self._cb_experience
        )
        self.create_subscription(
            OccupancyGrid, self.config.obstacle_map_topic,
            self._trainer_obstacle_callback, 10,
            callback_group=self._cb_experience
        )

        # Timeout de synchronisation : détecte les robots bloqués
        # Fix 5b : augmenté 120s → 400s.
        # Justification : batch_size=16 sur CPU → training step ~120-134s.
        # Pendant le calcul PyTorch, tous les robots sont CPU-starved et font
        # 0.01 steps/s. Avec 120s, le watchdog se déclenchait SYSTÉMATIQUEMENT,
        # resetant mineral_map=0 et bloquant make_decision en boucle infinie.
        # 400s = 1 épisode complet (300s) + 100s de marge → aucun faux positif.
        self._last_robot_time = {i: time.time() for i in range(self.config.num_robots)}
        self._sync_timeout = 400.0

        # Timers
        self.train_timer = self.create_timer(
            1.0, self.train_step_callback, callback_group=self._cb_train)
        self.epsilon_timer = self.create_timer(
            5.0, self.publish_epsilon, callback_group=self._cb_experience)
        self.save_timer = self.create_timer(
            60.0, self.save_checkpoint, callback_group=self._cb_train)
        self.watchdog_timer = self.create_timer(
            3.0, self._sync_watchdog, callback_group=self._cb_experience)

        # Racine du projet — recherche par marker (src/ + build/) pour être robuste
        # peu importe si Python tourne depuis src/, build/ ou install/.
        def _find_project_root():
            current = os.path.dirname(os.path.abspath(__file__))
            for _ in range(10):
                if (os.path.isdir(os.path.join(current, 'src')) and
                        os.path.isdir(os.path.join(current, 'build'))):
                    return current
                parent = os.path.dirname(current)
                if parent == current:
                    break
                current = parent
            # fallback: 3 niveaux (src layout)
            return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        _project_root = _find_project_root()

        # Dossiers de sauvegarde (dans le projet, pas dans ~/)
        self.save_dir = os.path.join(_project_root, 'models', 'qmix')
        os.makedirs(self.save_dir, exist_ok=True)

        # CSV logging
        self.log_dir = os.path.join(_project_root, 'logs', 'qmix')
        os.makedirs(self.log_dir, exist_ok=True)
        self._init_csv_loggers()

        # TensorBoard
        tb_dir = os.path.join(_project_root, 'logs', 'tensorboard')
        os.makedirs(tb_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=tb_dir)
        self.get_logger().info(f'TensorBoard: {tb_dir}')

        # Continuous learning
        self._load_checkpoint_if_exists()

        # C-3 — SIGTERM handler : sauvegarde le checkpoint avant que ROS2
        # ou le système ne tue le process. Évite de perdre les poids et
        # l'epsilon en cas d'arrêt forcé (systemd, OOM killer, ros2 lifecycle).
        signal.signal(signal.SIGTERM, self._on_sigterm)

        self.get_logger().info(
            f'QMIX Trainer initialized '
            f'[MultiScale={self.config.use_multi_scale}, '
            f'Comm={self.config.use_comm}, '
            f'CoverageWeight={self.config.coverage_weight}]'
        )

    # ══════════════════════════════════════════════════════════════════
    #  GESTION SIGNAUX
    # ══════════════════════════════════════════════════════════════════

    def _on_sigterm(self, signum, frame):
        """Sauvegarde d'urgence sur SIGTERM avant mort du process."""
        self.get_logger().warn(
            f'[C-3] SIGTERM reçu (train_step={self.train_step}, '
            f'ep={self.episode_count}, eps={self.epsilon:.4f}) '
            f'— sauvegarde checkpoint...'
        )
        try:
            self.save_checkpoint()
            self.writer.flush()
        except Exception as e:
            self.get_logger().error(f'SIGTERM save error: {e}')

    # ══════════════════════════════════════════════════════════════════
    #  INITIALISATION
    # ══════════════════════════════════════════════════════════════════

    def _init_networks(self):
        """Initialise les réseaux QMIX (avec multi-scale + comm)."""
        state_shape = self.config.state_shape

        self.qmix_network = QMixNetwork(
            state_shape=state_shape,
            num_agents=self.config.num_robots,
            num_actions=self.config.num_actions,
            hidden_dim=self.config.hidden_dim,
            hyper_dim=self.config.hyper_dim,
            local_obs_size=self.config.local_obs_size,
            use_multi_scale=self.config.use_multi_scale,
            use_comm=self.config.use_comm,
            comm_dim=self.config.comm_dim,
            num_comm_rounds=self.config.num_comm_rounds
        ).to(self.device)

        self.target_network = QMixTargetNetwork(
            QMixNetwork(
                state_shape=state_shape,
                num_agents=self.config.num_robots,
                num_actions=self.config.num_actions,
                hidden_dim=self.config.hidden_dim,
                hyper_dim=self.config.hyper_dim,
                local_obs_size=self.config.local_obs_size,
                use_multi_scale=self.config.use_multi_scale,
                use_comm=self.config.use_comm,
                comm_dim=self.config.comm_dim,
                num_comm_rounds=self.config.num_comm_rounds
            )
        ).to(self.device)
        self.target_network.update(self.qmix_network)

        # weight_decay=1e-4 : stoppe la croissance monotone des encoder/comm norms
        self.optimizer = optim.Adam(
            self.qmix_network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=1e-4
        )

        self.get_logger().info('Reseaux QMIX initialises (multi-scale + comm)')

    def _init_csv_loggers(self):
        self._episodes_csv = os.path.join(self.log_dir, 'episodes.csv')
        self._training_csv = os.path.join(self.log_dir, 'training.csv')

        if not os.path.exists(self._episodes_csv):
            with open(self._episodes_csv, 'w', newline='') as f:
                csv.writer(f).writerow([
                    'episode', 'steps', 'total_reward',
                    'avg_reward_per_step', 'epsilon',
                    'buffer_size', 'timestamp'
                ])

        if not os.path.exists(self._training_csv):
            with open(self._training_csv, 'w', newline='') as f:
                csv.writer(f).writerow([
                    'train_step', 'loss', 'epsilon', 'timestamp'
                ])

        self._eval_csv = os.path.join(self.log_dir, 'eval.csv')
        if not os.path.exists(self._eval_csv):
            with open(self._eval_csv, 'w', newline='') as f:
                csv.writer(f).writerow([
                    'eval_round', 'avg_reward', 'avg_reward_per_step',
                    'avg_steps', 'episode_count', 'train_step', 'timestamp'
                ])

        self.get_logger().info(f'CSV logs: {self.log_dir}')

    def _log_episode_csv(self, episode_data, total_reward):
        try:
            steps = len(episode_data['dones'])
            with open(self._episodes_csv, 'a', newline='') as f:
                csv.writer(f).writerow([
                    self.episode_count, steps,
                    round(total_reward, 3),
                    round(total_reward / max(steps, 1), 3),
                    round(self.epsilon, 4),
                    len(self.replay_buffer),
                    datetime.now().isoformat()
                ])
        except Exception as e:
            self.get_logger().error(f'CSV episode log error: {e}')

    def _log_training_csv(self, loss):
        try:
            with open(self._training_csv, 'a', newline='') as f:
                csv.writer(f).writerow([
                    self.train_step,
                    round(float(loss), 6),
                    round(self.epsilon, 4),
                    datetime.now().isoformat()
                ])
        except Exception as e:
            self.get_logger().error(f'CSV training log error: {e}')

    def _read_last_csv_episode(self):
        """Retourne le numéro du dernier épisode dans le CSV, ou 0 si absent."""
        try:
            if not os.path.exists(self._episodes_csv):
                return 0
            with open(self._episodes_csv, 'rb') as f:
                # Lecture efficace de la dernière ligne sans tout charger
                f.seek(0, 2)
                size = f.tell()
                if size == 0:
                    return 0
                f.seek(max(0, size - 256))
                last_bytes = f.read()
            last_line = last_bytes.decode('utf-8', errors='ignore').strip().split('\n')[-1]
            if not last_line or last_line.startswith('episode'):
                return 0
            return int(last_line.split(',')[0])
        except Exception:
            return 0

    def _load_checkpoint_if_exists(self):
        # Build candidate list: latest.pt first, then numbered checkpoints newest→oldest.
        # If latest.pt is corrupted (SIGKILL mid-write), we fall back automatically.
        latest_path = os.path.join(self.save_dir, 'latest.pt')
        numbered = sorted([
            os.path.join(self.save_dir, f)
            for f in os.listdir(self.save_dir)
            if f.startswith('checkpoint_') and f.endswith('.pt')
        ], reverse=True)
        candidates = ([latest_path] if os.path.exists(latest_path) else []) + numbered

        if not candidates:
            self.get_logger().info('No checkpoint found — starting from scratch')
            return

        checkpoint = None
        loaded_path = None
        for candidate in candidates:
            try:
                ck = torch.load(candidate, map_location=self.device, weights_only=True)
                # Sanity check: reject obviously corrupt/default checkpoints
                # (epsilon=1.0 AND train_step=0 means nothing was actually saved)
                if ck.get('train_step', 0) == 0 and abs(ck.get('epsilon', 1.0) - 1.0) < 1e-4:
                    self.get_logger().warn(
                        f'Checkpoint {os.path.basename(candidate)} appears corrupt '
                        f'(step=0, eps=1.0) — trying next'
                    )
                    continue
                checkpoint = ck
                loaded_path = candidate
                break
            except Exception as e:
                self.get_logger().warn(
                    f'Failed to load {os.path.basename(candidate)}: {e} — trying next'
                )

        if checkpoint is None:
            self.get_logger().error('All checkpoints failed or corrupt — starting from scratch')
            return

        try:
            self.qmix_network.load_state_dict(checkpoint['qmix_state_dict'])
            self.target_network.update(self.qmix_network)
            # FIX 1 — Reset Adam : les moments (m, v) accumulés depuis les crashs
            # poussent les gradients dans la mauvaise direction (loss 17→ explosion).
            # On garde les POIDS réseau et on repart avec un Adam frais.
            # La loss doit redescendre de ~17 à ~3-5 en quelques dizaines de steps.
            # Re-activer la ligne ci-dessous UNIQUEMENT après stabilisation de la loss
            # sous 5 pendant 50+ training steps consécutifs.
            if 'optimizer_state_dict' in checkpoint:
                self.get_logger().warn(
                    '[FIX-1] Optimizer Adam réinitialisé — momentum corrompu '
                    f'par crashs précédents ignoré (loss était ~17)'
                )
                # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.train_step = checkpoint.get('train_step', 0)
            self.episode_count = checkpoint.get('episode_count', 0)
            self.global_step = checkpoint.get('global_step', 0)
            self.epsilon = checkpoint.get('epsilon', self.config.epsilon_start)
            self.eval_round = checkpoint.get('eval_round', 0)

            # Lire la dernière ligne du CSV pour reprendre le compteur
            # d'épisodes depuis là où on s'est arrêté (le checkpoint peut
            # avoir 5 épisodes de retard sur le CSV)
            csv_last = self._read_last_csv_episode()
            if csv_last > self.episode_count:
                self.get_logger().info(
                    f'CSV ahead of checkpoint '
                    f'(csv={csv_last} > ckpt={self.episode_count}) '
                    f'— resuming from {csv_last}'
                )
                self.episode_count = csv_last

            self.get_logger().info(
                f'Checkpoint loaded from {os.path.basename(loaded_path)} '
                f'— train_step={self.train_step}, '
                f'episodes={self.episode_count}, '
                f'epsilon={self.epsilon:.4f}'
            )
            _eps_msg = Float32()
            _eps_msg.data = float(self.epsilon)
            self.epsilon_pub.publish(_eps_msg)
        except Exception as e:
            self.get_logger().error(
                f'Checkpoint load error: {e} — starting from scratch'
            )

        # Bug fix : _load_replay_buffer() appelée HORS du try/except du checkpoint.
        # Avant : si le pkl.gz échouait, l'exception était avalée par le bloc
        # extérieur et loguée comme "Erreur chargement checkpoint" — le buffer
        # restait vide silencieusement.
        # Maintenant : erreur replay buffer = log séparé, clairement identifiable.
        self._load_replay_buffer()

    # ══════════════════════════════════════════════════════════════════
    #  ÉPISODE BUFFER
    # ══════════════════════════════════════════════════════════════════

    def _init_episode_buffer(self):
        buf = {
            'mineral_maps': [[] for _ in range(self.config.num_robots)],
            'positions':    [[] for _ in range(self.config.num_robots)],
            'actions':      [[] for _ in range(self.config.num_robots)],
            'rewards':      [[] for _ in range(self.config.num_robots)],
            'global_states': [],
            'dones': [],
            'timestamps': [],
            # Composantes reward par robot (Priorité 4 — TensorBoard Reward/*)
            'reward_components': {i: None for i in range(self.config.num_robots)},
        }
        if self.config.use_multi_scale:
            buf['regional_maps'] = [[] for _ in range(self.config.num_robots)]
        return buf

    @staticmethod
    def _b64_to_arr(b64str, shape):
        """Décode un tableau base64 → numpy float32 avec la forme donnée."""
        return np.frombuffer(
            base64.b64decode(b64str), dtype=np.float32
        ).reshape(shape)

    def experience_callback(self, msg: String):
        """Callback pour recevoir les expériences des agents (arrays base64)."""
        try:
            data = json.loads(msg.data)
            robot_id = data['robot_id']
            step_data = data['step_data']

            obs_size = self.config.local_obs_size
            n_ch = self.config.state_shape[0]

            # Mise à jour du timestamp pour le watchdog (toujours, même si done)
            self._last_robot_time[robot_id] = time.time()

            # Ignorer les données d'un robot qui a déjà terminé son épisode
            # (évite de mélanger les données épisode N et N+1)
            if self._robot_done.get(robot_id, False):
                return

            # Observation locale (décodage base64) — float16 : 3.8 MB/ep au lieu de 7.7 MB/ep
            self.current_episode['mineral_maps'][robot_id].append(
                self._b64_to_arr(
                    step_data['mineral_map'], (obs_size, obs_size, n_ch)
                ).transpose(2, 0, 1).astype(np.float16)
            )
            self.current_episode['positions'][robot_id].append(
                step_data['position']
            )
            self.current_episode['actions'][robot_id].append(
                step_data['action']
            )

            # Reward = extrinsic + curiosity
            # reward inclut déjà le coverage bonus (calculé côté agent)
            r_ext = step_data['reward']
            self.current_episode['rewards'][robot_id].append(r_ext)

            # Composantes reward (envoyées avec le dernier step done=True)
            if 'reward_components' in step_data:
                self.current_episode['reward_components'][robot_id] = (
                    step_data['reward_components']
                )

            # Multi-scale : observation régionale (décodage base64) — float16 : 34 MB/ep au lieu de 69 MB/ep
            if (self.config.use_multi_scale and
                    'regional_map' in step_data and
                    'regional_maps' in self.current_episode):
                self.current_episode['regional_maps'][robot_id].append(
                    self._b64_to_arr(
                        step_data['regional_map'], (obs_size, obs_size, n_ch)
                    ).transpose(2, 0, 1).astype(np.float16)
                )

            # Fix G — détection done par step count (pas par flag done).
            #
            # Ancien garde : if step_data['done'] + robot_steps >= 50
            # Bug identifié : le message done=True arrive en DERNIÈRE position dans
            # la queue DDS (~position 1197/1200). Le trainer single-threaded bloque
            # 9-10s par training step (CPU). Entre deux training steps, seulement ~10
            # messages sont traités. Pour atteindre la position 1197 : ~1197s.
            # Le timeout DONE_WAITING des agents est 45s → ils auto-reset avant que
            # le trainer ne traite done=True. Résultat : stagnation à episodes=N.
            #
            # Nouveau garde : robot_steps >= 300 (sans dépendance au flag done)
            # Dès que robot_id a 300 steps dans le buffer courant, l'épisode est
            # terminé pour ce robot. Le message n°300 est en position ~1196 dans la
            # queue, mais les positions 1-299 le précèdent et sont traitées en ordre.
            # Immune aux backlogs : les messages done=True résiduels d'un ancien
            # épisode ont 0 steps dans le nouveau buffer → jamais déclenchés.
            robot_steps = len(self.current_episode['actions'][robot_id])
            if robot_steps >= 300:
                self._robot_done[robot_id] = True

            # Fix O-16 — Synchronisation par le robot le plus lent.
            # Ancien code : all(len > global_step) → un seul robot lent bloquait
            # global_step indéfiniment, causant des épisodes rejetés (n_dones trop bas).
            # Nouveau code : global_step avance jusqu'au min des robots (while loop).
            # Dès qu'un robot avance d'un pas, on enregistre le global_state.
            # Le robot lent drive la horloge — pas de blocage sur le plus rapide.
            min_steps = min(
                len(lst) for lst in self.current_episode['actions']
            )
            while self.global_step < min_steps:
                with self._map_lock:
                    # float16 : 36 MB/ep au lieu de 72 MB/ep (Fix J)
                    gs_snapshot = self._trainer_global_state.astype(np.float16)
                self.current_episode['global_states'].append(gs_snapshot)
                self.current_episode['dones'].append(step_data['done'])
                self.current_episode['timestamps'].append(time.time())
                self.global_step += 1

            # Fin d'épisode quand TOUS les robots ont envoyé done=True
            # ET que chaque robot a contribué au moins 50 pas dans ce buffer.
            # On utilise min(len(actions[i])) au lieu de global_step car
            # global_step se bloque facilement quand l'executor est occupé
            # à entraîner (single-threaded) et désynchronise les robots.
            if all(self._robot_done.values()):
                self._end_episode()

        except Exception as e:
            self.get_logger().error(f'Error processing experience: {e}')

    def _trainer_mineral_callback(self, msg: Float32MultiArray):
        """Construit et maintient le global_state du trainer depuis /mineral_map."""
        data = np.array(msg.data, dtype=np.float32)
        if data.size == self.config.map_height * self.config.map_width * 4:
            self._trainer_mineral_map = data.reshape(
                self.config.map_height, self.config.map_width, 4
            )
            self._rebuild_trainer_global_state()

    def _trainer_obstacle_callback(self, msg: OccupancyGrid):
        """Met à jour la couche obstacle du global_state du trainer."""
        data = np.array(msg.data, dtype=np.float32)
        if data.size == self.config.map_height * self.config.map_width:
            self._trainer_obstacle_map = data.reshape(
                self.config.map_height, self.config.map_width
            )
            self._rebuild_trainer_global_state()

    def _rebuild_trainer_global_state(self):
        # Fix I (O-10) — snapshot atomique de _trainer_global_state
        # Empêche experience_callback (Group A) de lire un état partiellement
        # construit si un futur refactor déplace ce code dans un autre groupe.
        with self._map_lock:
            mineral_ch  = self._trainer_mineral_map.transpose(2, 0, 1)   # (4, H, W)
            obstacle_ch = (self._trainer_obstacle_map > 0).astype(np.float32)[np.newaxis]
            # Le trainer n'a pas de carte d'exploration propre — on met zéros
            explore_ch  = np.zeros(
                (1, self.config.map_height, self.config.map_width), dtype=np.float32
            )
            self._trainer_global_state = np.concatenate(
                [mineral_ch, obstacle_ch, explore_ch], axis=0
            )

    def _sync_watchdog(self):
        """Détecte les robots bloqués et reset l'épisode pour débloquer."""
        # Ne rien surveiller si l'épisode n'a pas encore commencé
        if not any(self.current_episode['actions'][i]
                   for i in range(self.config.num_robots)):
            return

        now = time.time()
        for i in range(self.config.num_robots):
            # Un robot est bloqué s'il n'a pas envoyé de données depuis longtemps
            delay = now - self._last_robot_time.get(i, now)
            if delay > self._sync_timeout:
                self.get_logger().warn(
                    f'Robot {i} bloqué depuis {delay:.1f}s '
                    f'(sync_step={self.global_step}) — reset épisode'
                )
                self.current_episode = self._init_episode_buffer()
                self.global_step = 0
                self._robot_done = {j: False for j in range(self.config.num_robots)}
                self._last_robot_time = {
                    j: now for j in range(self.config.num_robots)
                }
                self._episode_start_time = now
                # Notifier le serveur et les agents pour redémarrer l'épisode
                reset_msg = String()
                reset_msg.data = json.dumps({
                    'episode': self.episode_count,
                    'watchdog': True,
                    'timestamp': time.time()
                })
                self.episode_reset_pub.publish(reset_msg)
                return

        # Step 4 — épisode bloqué : aucun robot individuel n'est silencieux (robots actifs)
        # MAIS l'épisode dure plus de 2× sa durée théorique (300s vs ~150s théorique).
        # Causes possibles : _end_episode() excepté silencieusement, _robot_done bloqué,
        # ou logic bug. Force le reset pour débloquer la boucle d'entraînement.
        _EPISODE_STALL_TIMEOUT = 300.0
        _episode_duration = now - self._episode_start_time
        if _episode_duration > _EPISODE_STALL_TIMEOUT and self.global_step > 50:
            _mx_free = self._end_episode_mutex.acquire(blocking=False)
            if _mx_free:
                self._end_episode_mutex.release()
            self.get_logger().error(
                f'[WATCHDOG-STALL] Épisode bloqué depuis {_episode_duration:.0f}s '
                f'(global_step={self.global_step}, ep={self.episode_count}, '
                f'mutex_locked={not _mx_free}) — force reset'
            )
            self.current_episode = self._init_episode_buffer()
            self.global_step = 0
            self._robot_done = {j: False for j in range(self.config.num_robots)}
            self._last_robot_time = {j: now for j in range(self.config.num_robots)}
            self._episode_start_time = now
            stall_msg = String()
            stall_msg.data = json.dumps({
                'episode': self.episode_count,
                'watchdog': True,
                'stall': True,
                'duration': round(_episode_duration, 1),
                'timestamp': time.time()
            })
            self.episode_reset_pub.publish(stall_msg)

    def _end_episode(self):
        # Fix I (O-2) — garde non-réentrant : _end_episode() s'exécute au plus
        # une fois simultanément. acquire(blocking=False) retourne False si le
        # verrou est déjà tenu → doublon rejeté immédiatement, pas de deadlock.
        # En fonctionnement normal (_cb_experience sérialisé), ce cas n'arrive
        # jamais. Le verrou est defense-in-depth + détection de régression.
        if not self._end_episode_mutex.acquire(blocking=False):
            self.get_logger().warn(
                f'[O-2] _end_episode() doublon rejeté '
                f'(thread={threading.get_ident()}, ep={self.episode_count})'
            )
            return

        # Structure plate : un seul try/except/finally.
        # Le finally garantit à la fois le reset des agents (Fix G / Change 3)
        # ET la libération du mutex (Fix I / O-2), y compris sur les early
        # returns (min_received, Fix H) et les exceptions.
        _reset_published = False
        try:
            # Snapshot + reset immédiat — même en cas d'exception plus bas,
            # le buffer est propre et _robot_done est False pour tous.
            # Sans ça, une exception laisserait _robot_done=True pour tous,
            # déclenchant _end_episode() en boucle à chaque nouvelle expérience.
            episode_snapshot = self.current_episode
            self.current_episode = self._init_episode_buffer()
            self.global_step = 0
            self._robot_done = {i: False for i in range(self.config.num_robots)}
            _now = time.time()
            self._last_robot_time = {i: _now for i in range(self.config.num_robots)}
            self._episode_start_time = _now

            # Rejeter les épisodes trop courts (messages DDS droppés malgré la
            # grande queue : couverture défensive, valeur min très basse = 10).
            min_received = min(
                len(episode_snapshot['mineral_maps'][i])
                for i in range(self.config.num_robots)
            )
            if min_received < 10:
                self.get_logger().warn(
                    f'Épisode rejeté : seulement {min_received} steps reçus '
                    f'(messages DDS droppés ?) — buffer ignoré'
                )
                return

            # Fix H — Rejeter les épisodes avec global_step incohérent
            # Problème : au restart depuis checkpoint, la queue DDS contient ~300
            # messages/robot de la session précédente. Ces messages remplissent
            # mineral_maps (min_received=300, check ci-dessus passé), MAIS le
            # global_step (= len(dones)) n'a que 6 entrées car les robots ont à
            # peine démarré. Résultat : épisode corrompu (Steps=6, Reward=9561)
            # qui entre dans le replay buffer et fait exploser GradNorm (~1703).
            # Ce check vérifie la cohérence entre dones et mineral_maps.
            n_dones = len(episode_snapshot['dones'])
            if n_dones < 50:
                self.get_logger().warn(
                    f'Épisode rejeté (Fix H) : global_step={n_dones} < 50 '
                    f'alors que mineral_maps={min_received} steps — '
                    f'contamination DDS probable au restart depuis checkpoint'
                )
                return
            if n_dones < min_received // 4:
                self.get_logger().warn(
                    f'Épisode rejeté (Fix H) : incohérence dones={n_dones} vs '
                    f'mineral_maps={min_received} (ratio < 0.25) — '
                    f'épisode probablement corrompu'
                )
                return

            episode_data = {
                'mineral_maps': [
                    np.stack(maps)
                    for maps in episode_snapshot['mineral_maps']
                ],
                'positions': [
                    np.stack(pos)
                    for pos in episode_snapshot['positions']
                ],
                'actions': [
                    np.array(act)
                    for act in episode_snapshot['actions']
                ],
                'rewards': [
                    np.array(rew)
                    for rew in episode_snapshot['rewards']
                ],
                'global_states': np.stack(
                    episode_snapshot['global_states']
                ),
                'dones': np.array(episode_snapshot['dones'])
            }

            # Multi-scale
            if (self.config.use_multi_scale and
                    'regional_maps' in episode_snapshot and
                    all(len(r) > 0
                        for r in episode_snapshot['regional_maps'])):
                episode_data['regional_maps'] = [
                    np.stack(reg)
                    for reg in episode_snapshot['regional_maps']
                ]

            steps = len(episode_data['dones'])
            total_reward = sum(sum(rew) for rew in episode_data['rewards'])

            # === MODE ÉVALUATION ===
            if self.is_eval_mode:
                self.eval_rewards_accum.append(total_reward)
                self.eval_steps_accum.append(steps)
                self.eval_episode_count += 1
                self.get_logger().info(
                    f'Eval episode {self.eval_episode_count}/'
                    f'{self.eval_episodes} - '
                    f'Reward: {total_reward:.1f}, Steps: {steps}'
                )
                if self.eval_episode_count >= self.eval_episodes:
                    self._end_eval_mode()
                # Nouvelle carte pour l'épisode suivant
                reset_msg = String()
                reset_msg.data = json.dumps({
                    'episode': self.episode_count,
                    'eval': True,
                    'timestamp': time.time()
                })
                self.episode_reset_pub.publish(reset_msg)
                _reset_published = True
                return  # buffer + _robot_done déjà réinitialisés en haut de _end_episode

            # === MODE ENTRAÎNEMENT ===
            # FIX 2 — Exclure les épisodes catastrophiques du buffer en temps réel.
            # Seuil identique à _load_replay_buffer() : reward < -400.
            _MIN_BUFFER_REWARD = -400.0
            if total_reward >= _MIN_BUFFER_REWARD:
                with self._replay_lock:
                    self.replay_buffer.add_episode(episode_data)
            else:
                self.get_logger().warn(
                    f'[FIX-2] Épisode exclu du buffer '
                    f'(reward={total_reward:.0f} < {_MIN_BUFFER_REWARD}) '
                    f'— boucle de mort évitée'
                )
            self.episode_count += 1
            self._last_episode_complete_time = time.time()

            self._log_episode_csv(episode_data, total_reward)

            self._reward_window.append(total_reward)
            ma50 = float(np.mean(self._reward_window))

            robot_rewards = [float(np.sum(rew)) for rew in episode_data['rewards']]
            minerals_per_robot = [
                int(np.sum(rew > 20)) for rew in episode_data['rewards']
            ]
            total_minerals = sum(minerals_per_robot)

            # TensorBoard — métriques générales
            ep = self.episode_count
            # self._minerals_window.append(total_minerals)
            self.writer.add_scalar('Episode/TotalReward',      total_reward, ep)
            self.writer.add_scalar('Episode/TotalReward_MA50', ma50,         ep)
            # self.writer.add_scalar('Episode/AvgReward_MA50', ma50 / max(steps, 1), ep)
            self.writer.add_scalar('Episode/Steps',    steps,                          ep)
            self.writer.add_scalar('Episode/Epsilon',  self.epsilon,                   ep)
            self.writer.add_scalar('Episode/BufferSize', len(self.replay_buffer),      ep)
            self.writer.add_scalar('Episode/MineralsDetected',      total_minerals,    ep)
            # self.writer.add_scalar('Episode/MineralsDetected_MA50',
            #                        float(np.mean(self._minerals_window)), ep)
            for i, (rr, mn) in enumerate(zip(robot_rewards, minerals_per_robot)):
                # self._robot_r_windows[i].append(rr)
                # self._robot_m_windows[i].append(mn)
                self.writer.add_scalar(f'Robots/Robot{i}_Reward',   rr, ep)
                # self.writer.add_scalar(f'Robots/Robot{i}_Reward_MA50',
                #                        float(np.mean(self._robot_r_windows[i])), ep)
                self.writer.add_scalar(f'Robots/Robot{i}_Minerals', mn, ep)
                # self.writer.add_scalar(f'Robots/Robot{i}_Minerals_MA50',
                #                        float(np.mean(self._robot_m_windows[i])), ep)

            # ── TensorBoard — Reward components (Priorité 4) ──────────
            rc = episode_snapshot.get('reward_components', {})
            if all(rc.get(i) is not None for i in range(self.config.num_robots)):
                total_mineral  = sum(rc[i]['mineral']  for i in range(self.config.num_robots))
                total_coverage = sum(rc[i]['coverage'] for i in range(self.config.num_robots))
                total_penalty  = sum(rc[i]['penalty']  for i in range(self.config.num_robots))
                # self._mineral_r_window.append(total_mineral)
                # self._coverage_window.append(total_coverage)
                # self._penalty_window.append(total_penalty)
                self.writer.add_scalar('Reward/Mineral',  total_mineral,  ep)
                self.writer.add_scalar('Reward/Coverage', total_coverage, ep)
                self.writer.add_scalar('Reward/Penalty',  total_penalty,  ep)
                # self.writer.add_scalar('Reward/Mineral_MA50',
                #                        float(np.mean(self._mineral_r_window)), ep)
                # self.writer.add_scalar('Reward/Coverage_MA50',
                #                        float(np.mean(self._coverage_window)), ep)
                # self.writer.add_scalar('Reward/Penalty_MA50',
                #                        float(np.mean(self._penalty_window)), ep)

            # ── TensorBoard — CONTRIBUTION 2 : Communication ──────────
            if self.config.use_comm:
                try:
                    with torch.no_grad():
                        comm = self.qmix_network.shared_agent_network.comm_module
                        gate_weights = comm.gate[0].weight
                        gate_bias = comm.gate[0].bias
                        self.writer.add_scalar(
                            'Comm/GateWeightNorm', gate_weights.norm().item(), ep
                        )
                        self.writer.add_scalar(
                            'Comm/GateBiasMean', gate_bias.mean().item(), ep
                        )
                except Exception:
                    pass

            # ── TensorBoard — CONTRIBUTION 2 : Multi-Scale ────────────
            if self.config.use_multi_scale:
                try:
                    with torch.no_grad():
                        ms_enc = self.qmix_network.shared_agent_network.encoder
                        local_norm = sum(
                            p.norm().item()
                            for p in ms_enc.local_encoder.parameters()
                        )
                        regional_norm = sum(
                            p.norm().item()
                            for p in ms_enc.regional_encoder.parameters()
                        )
                        self.writer.add_scalar(
                            'MultiScale/LocalEncoderNorm', local_norm, ep
                        )
                        self.writer.add_scalar(
                            'MultiScale/RegionalEncoderNorm', regional_norm, ep
                        )
                        # Cross-attention weights norm
                        attn_norm = sum(
                            p.norm().item()
                            for p in ms_enc.cross_attention.parameters()
                        )
                        self.writer.add_scalar(
                            'MultiScale/CrossAttentionNorm', attn_norm, ep
                        )
                except Exception:
                    pass

            self._session_episodes += 1

            self.get_logger().info(
                f'Episode {self.episode_count} - '
                f'Steps: {steps}, Reward: {total_reward:.1f}, '
                f'Minerals: {total_minerals} {minerals_per_robot}'
            )

            # Sauvegarde checkpoint tous les 5 épisodes DE LA SESSION COURANTE.
            # Bug fix : on utilise _session_episodes (repart à 0 au restart) et
            # non episode_count (chargé du CSV). Évite le save prématuré quand
            # episode_count % 5 == 0 sur le 1er épisode (ex: ep 85, buffer=1).
            if self._session_episodes % 5 == 0:
                self.save_checkpoint()

            if self.episode_count % self.eval_freq == 0:
                self._start_eval_mode()

            # Signaler au serveur de régénérer la carte minérale
            reset_msg = String()
            reset_msg.data = json.dumps({
                'episode': self.episode_count,
                'timestamp': time.time()
            })
            self.episode_reset_pub.publish(reset_msg)
            _reset_published = True
            # buffer + _robot_done déjà réinitialisés en haut de _end_episode

        except Exception as e:
            self.get_logger().error(
                f'Error ending episode (ep={self.episode_count}): {e}\n'
                f'{traceback.format_exc()}'
            )
        finally:
            # Fix G (Change 3) + Fix I (O-2) — dans TOUS les cas :
            # 1. Publier le reset si pas encore fait (agents attendent 45s sinon)
            # 2. Libérer le mutex (couvre early returns Fix H + exceptions)
            # IMPORTANT : publish() est dans son propre try/except pour garantir
            # que release() est TOUJOURS atteint même si DDS est indisponible.
            # Sans ce guard, une exception publish() verrouille le mutex à jamais.
            if not _reset_published:
                try:
                    _fb = String()
                    _fb.data = json.dumps({
                        'episode': self.episode_count,
                        'timestamp': time.time()
                    })
                    self.episode_reset_pub.publish(_fb)
                    self.get_logger().warn(
                        f'[finally] reset publié ep={self.episode_count} '
                        f'(exception dans _end_episode)'
                    )
                except Exception as _pub_err:
                    self.get_logger().error(
                        f'[CRITICAL] finally publish failed: {_pub_err} — '
                        f"agents bloqués jusqu'au watchdog (45s)"
                    )
            self._end_episode_mutex.release()

    # ══════════════════════════════════════════════════════════════════
    #  ÉVALUATION
    # ══════════════════════════════════════════════════════════════════

    def _start_eval_mode(self):
        self.is_eval_mode = True
        self.eval_episode_count = 0
        self.eval_rewards_accum = []
        self.eval_steps_accum = []
        self.eval_round += 1
        msg = Float32()
        msg.data = 0.0
        self.epsilon_pub.publish(msg)
        self.get_logger().info(
            f'=== EVALUATION #{self.eval_round} demarree '
            f'({self.eval_episodes} episodes, eps=0.0) ==='
        )

    def _end_eval_mode(self):
        avg_reward = float(np.mean(self.eval_rewards_accum))
        avg_steps = float(np.mean(self.eval_steps_accum))
        avg_rps = float(np.mean([
            r / max(s, 1)
            for r, s in zip(self.eval_rewards_accum, self.eval_steps_accum)
        ]))

        try:
            with open(self._eval_csv, 'a', newline='') as f:
                csv.writer(f).writerow([
                    self.eval_round, round(avg_reward, 3),
                    round(avg_rps, 4), round(avg_steps, 1),
                    self.episode_count, self.train_step,
                    datetime.now().isoformat()
                ])
        except Exception as e:
            self.get_logger().error(f'CSV eval log error: {e}')

        # self._eval_window.append(avg_reward)
        # self._eval_rps_window.append(avg_rps)
        self.writer.add_scalar('Eval/AvgReward',        avg_reward, self.eval_round)
        # self.writer.add_scalar('Eval/AvgReward_MA50',
        #                        float(np.mean(self._eval_window)), self.eval_round)
        self.writer.add_scalar('Eval/AvgRewardPerStep', avg_rps,    self.eval_round)
        # self.writer.add_scalar('Eval/AvgRewardPerStep_MA50',
        #                        float(np.mean(self._eval_rps_window)), self.eval_round)
        self.writer.add_scalar('Eval/AvgSteps',         avg_steps,  self.eval_round)

        self.get_logger().info(
            f'=== EVALUATION #{self.eval_round} terminee === '
            f'Reward moy: {avg_reward:.1f}, '
            f'Reward/step: {avg_rps:.3f}'
        )
        self.is_eval_mode = False

    # ══════════════════════════════════════════════════════════════════
    #  ENTRAÎNEMENT
    # ══════════════════════════════════════════════════════════════════

    def train_step_callback(self):
        """Effectue un pas d'entraînement QMIX."""
        # Attendre d'avoir au moins 1 épisode complet dans le buffer
        if len(self.replay_buffer) < 1:
            return

        try:
            with self._replay_lock:
                batch = self.replay_buffer.sample_batch()

            # === 1. Loss QMIX (avec multi-scale + comm) ===
            loss, q_tot_mean, q_tot_std = self._compute_loss(batch)

            # Vérification NaN avant backward — évite de corrompre les poids
            if not torch.isfinite(loss):
                self.get_logger().warn(
                    f'Loss non-finie ({loss.item():.4f}) — skip training step'
                )
                return

            self.optimizer.zero_grad()
            loss.backward()

            grad_norm = 0.0
            if self.config.grad_clip > 0:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(
                    self.qmix_network.parameters(),
                    self.config.grad_clip  # config.grad_clip devrait être 1.0
                ))
            self.optimizer.step()

            self.train_step += 1

            # C-1 — Decay epsilon par pas d'entraînement (pas recomputation).
            # Avant : publish_epsilon() recalculait depuis epsilon_start chaque
            # 5s → si le checkpoint chargé avait epsilon=0.05 mais train_step
            # faible, epsilon remontait vers 1.0 au premier appel timer.
            # Maintenant : décroissance incrémentale depuis la valeur courante
            # → l'epsilon chargé du checkpoint est TOUJOURS respecté.
            _eps_decay = (
                (self.config.epsilon_start - self.config.epsilon_end)
                / max(1, self.config.epsilon_decay)
            )
            self.epsilon = max(self.config.epsilon_end,
                               self.epsilon - _eps_decay)

            # Target network update
            if self.train_step % self.config.target_update_freq == 0:
                self.target_network.update(self.qmix_network)
                self.get_logger().info(
                    f'Target network updated at step {self.train_step}'
                )

            # Logging
            self._log_training_csv(loss.item())

            ts = self.train_step
            # self._loss_window.append(loss.item())
            # self._gradnorm_window.append(grad_norm)
            # self._qtot_window.append(q_tot_mean)
            self.writer.add_scalar('Train/Loss',      loss.item(), ts)
            # self.writer.add_scalar('Train/Loss_MA50', float(np.mean(self._loss_window)), ts)
            self.writer.add_scalar('Train/Epsilon',   self.epsilon, ts)
            self.writer.add_scalar('Train/GradNorm',  grad_norm,   ts)
            # self.writer.add_scalar('Train/GradNorm_MA50', float(np.mean(self._gradnorm_window)), ts)
            self.writer.add_scalar('Train/QTot_Mean', q_tot_mean,  ts)
            # self.writer.add_scalar('Train/QTot_Mean_MA50', float(np.mean(self._qtot_window)), ts)
            self.writer.add_scalar('Train/QTot_Std',  q_tot_std,   ts)

            if self.train_step % 10 == 0:
                self.get_logger().info(
                    f'Train step {self.train_step} - '
                    f'Loss: {loss.item():.4f}, '
                    f'QTot: {q_tot_mean:.1f}+/-{q_tot_std:.1f}, '
                    f'GradNorm: {grad_norm:.2f}'
                )

            if self.train_step % 50 == 0:
                self._publish_network_weights()
                self._log_cnn_stats(batch, ts)

        except Exception as e:
            self.get_logger().error(f'Training error: {e}')
            import traceback
            traceback.print_exc()

    def _compute_loss(self, batch):
        """
        Loss TD(n) QMIX — forward pass VECTORISÉ sur tous les timesteps.

        Avant : T-1 forward passes séquentiels (boucle for t in range(T-1))
        Après : 1 forward pass online + 1 forward pass target sur (T*B) éléments
        Gain  : ~T× plus rapide (T ≈ 300 timesteps par épisode)
        """
        T = batch['global_states'].size(0)
        T = min(T, 64)  # Cap à 64 steps : évite les blocks CPU de 150s+ (T≈300 → ~30s)
        B = batch['global_states'].size(1)
        TB = T * B

        regional_maps = batch.get('regional_maps', None)

        # ── 0. Slicer tous les tensors à T avant reshape (T < T_max du buffer) ─
        mineral_maps_t = [maps[:T] for maps in batch['mineral_maps']]
        positions_t    = [pos[:T]  for pos  in batch['positions']]
        gs_t           = batch['global_states'][:T]
        actions_t      = [a[:T]    for a    in batch['actions']]
        rewards_t      = [r[:T]    for r    in batch['rewards']]
        dones_t        = batch['dones'][:T]
        regional_t     = ([reg[:T] for reg in regional_maps]
                          if regional_maps is not None else None)

        # ── 1. Aplatir (T,B,...) → (T*B,...) pour forward pass unique ────
        maps_flat = [maps.reshape(TB, *maps.shape[2:])
                     for maps in mineral_maps_t]
        pos_flat  = [pos.reshape(TB, *pos.shape[2:])
                     for pos in positions_t]
        gs_flat   = gs_t.reshape(TB, *gs_t.shape[2:])
        regs_flat = ([reg.reshape(TB, *reg.shape[2:]) for reg in regional_t]
                     if regional_t is not None else None)

        # ── 2. Forward online — 1 seul appel réseau ───────────────────────
        _, q_ind_flat = self.qmix_network(
            maps_flat, pos_flat, gs_flat, regional_maps=regs_flat
        )
        q_ind = [q.view(T, B, -1) for q in q_ind_flat]  # list[N] (T,B,A)

        # ── 3. Forward target — 1 seul appel, sans gradient ──────────────
        with torch.no_grad():
            _, tgt_flat = self.target_network(
                maps_flat, pos_flat, gs_flat, regional_maps=regs_flat
            )
            tgt_q_max_flat = torch.stack(
                [q.max(1)[0] for q in tgt_flat], dim=1
            )  # (T*B, N)
            gs_enc_tgt = self.target_network.qmix_network.encode_state(gs_flat)
            q_tot_tgt = self.target_network.qmix_network.mixing_network(
                tgt_q_max_flat, gs_enc_tgt
            ).view(T, B)  # (T, B)

        # ── 4. Actions et masque ──────────────────────────────────────────
        actions = actions_t  # list[N] of (T, B) — déjà slicé à T
        actions_clamped = [a.clamp(min=0) for a in actions]
        valid = torch.stack(
            [a >= 0 for a in actions], dim=2
        ).all(dim=2)  # (T, B)

        # Q-values choisies : (T, B, N)
        chosen_qs = torch.stack([
            q_ind[i].gather(2, actions_clamped[i].unsqueeze(2)).squeeze(2)
            for i in range(self.config.num_robots)
        ], dim=2)

        # ── 5. Mixing vectorisé ───────────────────────────────────────────
        gs_enc_online = self.qmix_network.encode_state(gs_flat)  # (T*B, 256)
        q_tot = self.qmix_network.mixing_network(
            chosen_qs.reshape(TB, self.config.num_robots), gs_enc_online
        ).view(T, B)  # (T, B)

        # ── 6. Cibles TD(n) — boucle légère sur scalaires précalculés ────
        rewards = torch.stack(rewards_t, dim=2).sum(dim=2)  # (T, B)
        dones   = dones_t.float()                            # (T, B)

        # Clipping per-step pour éliminer les pics extrêmes avant normalisation.
        # Valeurs choisies d'après les stats CSV : total/step ∈ [-50, 200]
        # couvre 99%+ des épisodes observés sans couper le signal utile.
        rewards = rewards.clamp(-50.0, 200.0)

        # Normalisation des récompenses — critique pour éviter l'explosion
        # de gradient quand la variance inter-agents est extrême (ex: robot 3
        # = 835 reward vs robot 2 = 35, ratio 24×).
        # Normalise vers mean≈0, std≈1 sur tout le batch.
        r_mean = rewards.mean()
        r_std  = rewards.std() + 1e-8
        rewards = (rewards - r_mean) / r_std
        n       = self.config.n_steps
        gamma   = self.config.gamma

        targets = torch.zeros(T - 1, B, device=self.device)
        for t in range(T - 1):
            ret      = torch.zeros(B, device=self.device)
            not_done = torch.ones(B, device=self.device)
            discount = 1.0
            for k in range(n):
                tk = t + k
                if tk >= T - 1:
                    break
                ret      = ret + discount * rewards[tk] * not_done
                not_done = not_done * (1.0 - dones[tk])
                discount *= gamma
            bootstrap_t = t + n
            if bootstrap_t < T:
                with torch.no_grad():
                    ret = ret + discount * q_tot_tgt[bootstrap_t] * not_done
            targets[t] = ret

        # ── 7. TD loss ────────────────────────────────────────────────────
        mask     = valid[:T - 1].float()  # (T-1, B)
        td_error = F.smooth_l1_loss(
            q_tot[:T - 1], targets.detach(), reduction='none'
        )
        loss = (td_error * mask).sum() / mask.sum().clamp(min=1)

        with torch.no_grad():
            q_valid    = q_tot[:T - 1][valid[:T - 1]]
            q_tot_mean = float(q_valid.mean()) if q_valid.numel() > 0 else 0.0
            q_tot_std  = float(q_valid.std())  if q_valid.numel() > 1 else 0.0

        return loss, q_tot_mean, q_tot_std

    # ══════════════════════════════════════════════════════════════════
    #  LOGGING & PUBLICATION
    # ══════════════════════════════════════════════════════════════════

    def _log_cnn_stats(self, batch, step):
        """Histogrammes CNN + feature maps dans TensorBoard."""
        encoder = self.qmix_network.shared_agent_network.encoder

        # Accéder au bon encodeur selon multi-scale
        if self.config.use_multi_scale:
            actual_encoder = encoder.local_encoder
        else:
            actual_encoder = encoder

        with torch.no_grad():
            self.writer.add_histogram(
                'CNN/conv1_weights', actual_encoder.conv1.weight, step
            )
            self.writer.add_histogram(
                'CNN/conv2_weights', actual_encoder.conv2.weight, step
            )
            self.writer.add_histogram(
                'CNN/conv3_weights', actual_encoder.conv3.weight, step
            )
            self.writer.add_histogram(
                'CNN/fc_weights', actual_encoder.fc.weight, step
            )

        try:
            with torch.no_grad():
                sample_map = batch['mineral_maps'][0][0].unsqueeze(0)

                activations = {}
                def make_hook(name):
                    def hook(module, input, output):
                        activations[name] = output.detach().cpu()
                    return hook

                hooks = [
                    actual_encoder.conv1.register_forward_hook(
                        make_hook('conv1')
                    ),
                    actual_encoder.conv2.register_forward_hook(
                        make_hook('conv2')
                    ),
                ]
                actual_encoder(sample_map)
                for h in hooks:
                    h.remove()

                for name, acts in activations.items():
                    feat = acts[0].unsqueeze(1)
                    f_min = feat.flatten(1).min(1)[0].view(-1, 1, 1, 1)
                    f_max = feat.flatten(1).max(1)[0].view(-1, 1, 1, 1)
                    denom = (f_max - f_min).clamp(min=1e-6)
                    feat = (feat - f_min) / denom
                    self.writer.add_images(
                        f'CNN/{name}_feature_maps', feat, step
                    )

                # Communication gate stats
                if self.config.use_comm:
                    comm = self.qmix_network.shared_agent_network.comm_module
                    for name, param in comm.gate.named_parameters():
                        self.writer.add_histogram(
                            f'Comm/gate_{name}', param, step
                        )

        except Exception as e:
            self.get_logger().debug(f'CNN activation log skipped: {e}')

    def _publish_network_weights(self):
        """Publie les poids (QMIX + ICM) pour les agents."""
        try:
            save_path = '/tmp/qmix_weights.pt'
            data = {
                'state_dict': self.qmix_network.state_dict(),
                'train_step': self.train_step,
                'episode_count': self.episode_count
            }

            torch.save(data, save_path)

            msg = String()
            msg.data = json.dumps({
                'type': 'weight_update',
                'path': save_path,
                'train_step': self.train_step,
                'timestamp': time.time()
            })
            self.weight_update_pub.publish(msg)

        except Exception as e:
            self.get_logger().error(f'Error publishing weights: {e}')

    def publish_epsilon(self):
        # C-1 — Ne PAS recalculer epsilon ici.
        # La décroissance est faite dans train_step_callback (par pas d'entraînement).
        # Ce timer publie simplement la valeur courante pour les agents.
        msg = Float32()
        msg.data = 0.0 if self.is_eval_mode else float(self.epsilon)
        self.epsilon_pub.publish(msg)

    def _save_replay_buffer_bg(self):
        """Sauvegarde le replay buffer sur disque en thread background.

        - Limite aux 200 derniers épisodes pour ne pas saturer le disque.
        - compresslevel=3 : bon compromis vitesse/taille (gzip ~3-5x).
        - Écriture atomique via fichier .tmp + os.replace() pour éviter
          un fichier corrompu si le process est tué pendant la sauvegarde.
        """
        import gzip
        import pickle

        buf_path = os.path.join(self.save_dir, 'replay_buffer.pkl.gz')

        def _do_save():
            try:
                with self._replay_lock:
                    episodes = list(self.replay_buffer.episodes)[-200:]
                n = len(episodes)
                if n == 0:
                    return
                tmp = buf_path + '.tmp'
                with gzip.open(tmp, 'wb', compresslevel=3) as f:
                    pickle.dump(episodes, f)
                os.replace(tmp, buf_path)
                size_mb = os.path.getsize(buf_path) / 1e6
                self.get_logger().info(
                    f'Replay buffer sauvegardé ({n} épisodes, {size_mb:.0f} MB)'
                )
            except Exception as e:
                self.get_logger().error(f'Erreur save replay buffer: {e}')

        threading.Thread(target=_do_save, daemon=True).start()

    def _load_replay_buffer(self):
        """Charge le replay buffer depuis le disque au démarrage.

        Appelée HORS du try/except du checkpoint pour que les erreurs
        soient clairement identifiables dans les logs (pas confondues
        avec des erreurs de chargement des poids réseau).
        """
        import gzip
        import pickle
        import traceback

        buf_path = os.path.join(self.save_dir, 'replay_buffer.pkl.gz')
        if not os.path.exists(buf_path):
            self.get_logger().info('Replay buffer : aucun fichier trouvé — buffer vide')
            return
        try:
            with gzip.open(buf_path, 'rb') as f:
                episodes = pickle.load(f)

            if not isinstance(episodes, list):
                self.get_logger().error(
                    f'Replay buffer corrompu : type={type(episodes)} attendu list'
                )
                return

            required_keys = {'mineral_maps', 'positions', 'actions',
                             'rewards', 'global_states', 'dones'}
            # FIX 2 — Seuil de filtrage : épisodes catastrophiques exclus du buffer.
            # En dessous de ce seuil, la politique converge vers "foncer dans les murs"
            # (pénalité collision × N steps > tout le reste). Ces épisodes créent une
            # boucle de mort : mauvaise politique → mauvais épisodes → Q-values pires.
            _MIN_BUFFER_REWARD = -400.0
            loaded = 0
            skipped = 0
            filtered_bad = 0
            for ep in episodes:
                if not isinstance(ep, dict):
                    skipped += 1
                    continue
                if not required_keys.issubset(ep.keys()):
                    missing = required_keys - ep.keys()
                    self.get_logger().warn(
                        f'Épisode ignoré — clés manquantes: {missing}'
                    )
                    skipped += 1
                    continue
                # Filtrer les épisodes catastrophiques (boucle de mort)
                try:
                    ep_reward = sum(
                        float(np.sum(ep['rewards'][i]))
                        for i in range(len(ep['rewards']))
                    )
                    if ep_reward < _MIN_BUFFER_REWARD:
                        filtered_bad += 1
                        continue
                except Exception:
                    pass  # si calcul échoue, charger quand même
                self.replay_buffer.episodes.append(ep)
                loaded += 1

            self.get_logger().info(
                f'Replay buffer chargé : {loaded} épisodes restaurés'
                + (f', {skipped} ignorés (structure invalide)' if skipped else '')
                + (f', {filtered_bad} filtrés (reward < {_MIN_BUFFER_REWARD})' if filtered_bad else '')
            )

        except Exception as e:
            self.get_logger().error(
                f'Erreur load replay buffer: {e}\n{traceback.format_exc()}'
            )

    def save_checkpoint(self):
        """Sauvegarde un checkpoint QMIX."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = os.path.join(self.save_dir, f'checkpoint_{timestamp}.pt')
            latest_path = os.path.join(self.save_dir, 'latest.pt')

            data = {
                'qmix_state_dict': self.qmix_network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_step': self.train_step,
                'episode_count': self.episode_count,
                'global_step': self.global_step,
                'epsilon': self.epsilon,
                'eval_round': self.eval_round,
                'config': self.config.__dict__
            }

            # C-3 — Écriture atomique pour les deux fichiers.
            # Sans ça, un SIGKILL pendant torch.save() laisse un fichier corrompu.
            # os.replace() est atomique sur Linux (rename syscall).
            tmp_numbered = path + '.tmp'
            torch.save(data, tmp_numbered)
            os.replace(tmp_numbered, path)

            tmp_path = latest_path + '.tmp'
            torch.save(data, tmp_path)
            os.replace(tmp_path, latest_path)

            # Garder seulement les 3 derniers
            checkpoints = sorted([
                f for f in os.listdir(self.save_dir)
                if f.startswith('checkpoint_') and f.endswith('.pt')
            ])
            for old in checkpoints[:-3]:
                os.remove(os.path.join(self.save_dir, old))

            self.get_logger().info(
                f'Checkpoint saved (step={self.train_step}, '
                f'episodes={self.episode_count})'
            )
            # Replay buffer sauvegardé en parallèle (thread daemon)
            self._save_replay_buffer_bg()

        except Exception as e:
            self.get_logger().error(f'Error saving checkpoint: {e}')


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    node = QMIXTrainerNode()
    # Fix G (Change 2) — MultiThreadedExecutor : experience_callback tourne en
    # parallèle de train_step_callback (plus de blocage 9-10s sur la queue DDS)
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down QMIX Trainer...')
    finally:
        node.get_logger().info('Sauvegarde finale du checkpoint...')
        node.save_checkpoint()
        node.writer.flush()
        node.writer.close()
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
