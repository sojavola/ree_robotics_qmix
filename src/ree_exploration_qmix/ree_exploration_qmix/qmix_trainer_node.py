#!/usr/bin/env python3
"""
Noeud d'entraînement centralisé QMIX avec 3 contributions PhD :

  1. Geo-ICM : Entraînement du forward model (curiosité intrinsèque)
     → L_ICM = ||φ̂(s_{t+1}) - φ(s_{t+1})||²
     → Optimiseur séparé (Adam, lr=0.001)
     → Le forward model est partagé avec les agents via les checkpoints

  2. Multi-Scale : Le forward pass utilise regional_maps si disponibles
     → Pas de changement dans la loss QMIX
     → Le CNN multi-scale est entraîné end-to-end via la loss QMIX

  3. GeoCommQMIX : Communication dans le forward pass centralisé
     → Pendant l'entraînement, les messages sont recalculés à chaque pas
     → Le CommModule (gate, attention) est entraîné end-to-end via QMIX

Respect CTDE :
  - L'état global est utilisé UNIQUEMENT dans le mixing network
  - Les Q-values individuelles dépendent des observations locales + messages
  - La monotonie QMIX (IGM) est préservée (hypernetwork inchangé)
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String, Int32, Float32
from geometry_msgs.msg import Pose2D
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import os
import json
import base64
import time
import csv
from collections import deque
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from .networks import QMixNetwork, QMixTargetNetwork
from .replay_buffer import QMIXReplayBuffer
from .config import QMIXConfig
from .geo_icm import GeoICM


class QMIXTrainerNode(Node):
    """Noeud d'entraînement centralisé QMIX avec ICM + Multi-Scale + Comm."""

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

        # Initialiser Geo-ICM (Contribution 1)
        self._init_icm()

        # Replay buffer
        self.replay_buffer = QMIXReplayBuffer(
            buffer_size=self.config.buffer_size,
            batch_size=self.config.batch_size,
            num_robots=self.config.num_robots,
            device=self.device
        )

        # Buffer temporaire pour l'épisode en cours
        self.current_episode = self._init_episode_buffer()

        # Compteurs
        self.train_step = 0
        self.episode_count = 0
        self.global_step = 0
        self.epsilon = self.config.epsilon_start

        # Fenêtre glissante pour les moving averages
        self._reward_window = deque(maxlen=10)

        # === EVAL SPLIT ===
        self.eval_freq = 20
        self.eval_episodes = 1
        self.is_eval_mode = False
        self.eval_episode_count = 0
        self.eval_round = 0
        self.eval_rewards_accum = []
        self.eval_steps_accum = []

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

        # Subscribers
        self.exp_sub = self.create_subscription(
            String, self.config.agent_exp_topic,
            self.experience_callback, 10
        )

        # Timeout de synchronisation : détecte les robots bloqués
        self._last_robot_time = {i: time.time() for i in range(self.config.num_robots)}
        self._sync_timeout = 10.0  # secondes avant de considérer un robot bloqué

        # Timers
        self.train_timer = self.create_timer(1.0, self.train_step_callback)
        self.epsilon_timer = self.create_timer(5.0, self.publish_epsilon)
        self.save_timer = self.create_timer(60.0, self.save_checkpoint)
        self.watchdog_timer = self.create_timer(3.0, self._sync_watchdog)

        # Racine du projet (3 niveaux au-dessus de ce fichier)
        _project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..', '..')
        )

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

        self.get_logger().info(
            f'QMIX Trainer initialized '
            f'[ICM={self.config.use_icm}, '
            f'MultiScale={self.config.use_multi_scale}, '
            f'Comm={self.config.use_comm}]'
        )

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

        self.optimizer = optim.Adam(
            self.qmix_network.parameters(),
            lr=self.config.learning_rate
        )

        self.get_logger().info('Reseaux QMIX initialises (multi-scale + comm)')

    def _init_icm(self):
        """Initialise le module Geo-ICM."""
        if not self.config.use_icm:
            self.icm = None
            self.icm_optimizer = None
            return

        self.icm = GeoICM(
            feature_dim=self.config.icm_feature_dim,
            action_dim=self.config.num_actions,
            hidden_dim=self.config.icm_hidden_dim,
            curiosity_weight=self.config.curiosity_weight
        ).to(self.device)

        self.icm_optimizer = optim.Adam(
            self.icm.parameters(),
            lr=self.config.icm_lr
        )

        self.get_logger().info(
            f'Geo-ICM initialise (beta={self.config.curiosity_weight}, '
            f'lr={self.config.icm_lr})'
        )

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
                    'train_step', 'loss', 'icm_loss', 'epsilon', 'timestamp'
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

    def _log_training_csv(self, loss, icm_loss=0.0):
        try:
            with open(self._training_csv, 'a', newline='') as f:
                csv.writer(f).writerow([
                    self.train_step,
                    round(float(loss), 6),
                    round(float(icm_loss), 6),
                    round(self.epsilon, 4),
                    datetime.now().isoformat()
                ])
        except Exception as e:
            self.get_logger().error(f'CSV training log error: {e}')

    def _load_checkpoint_if_exists(self):
        latest_path = os.path.join(self.save_dir, 'latest.pt')
        if not os.path.exists(latest_path):
            self.get_logger().info('Aucun checkpoint trouve — demarrage a zero')
            return
        try:
            checkpoint = torch.load(latest_path, map_location=self.device, weights_only=True)
            self.qmix_network.load_state_dict(checkpoint['qmix_state_dict'])
            self.target_network.update(self.qmix_network)
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.train_step = checkpoint.get('train_step', 0)
            self.episode_count = checkpoint.get('episode_count', 0)
            self.global_step = checkpoint.get('global_step', 0)
            self.epsilon = checkpoint.get('epsilon', self.config.epsilon_start)
            self.eval_round = checkpoint.get('eval_round', 0)

            # ICM checkpoint
            if self.config.use_icm and 'icm_state_dict' in checkpoint:
                self.icm.load_state_dict(checkpoint['icm_state_dict'])
                if 'icm_optimizer_state_dict' in checkpoint:
                    self.icm_optimizer.load_state_dict(
                        checkpoint['icm_optimizer_state_dict']
                    )

            self.get_logger().info(
                f'Checkpoint charge — train_step={self.train_step}, '
                f'episodes={self.episode_count}, '
                f'epsilon={self.epsilon:.3f}'
            )
        except Exception as e:
            self.get_logger().error(
                f'Erreur chargement checkpoint: {e} — demarrage a zero'
            )

    # ══════════════════════════════════════════════════════════════════
    #  ÉPISODE BUFFER
    # ══════════════════════════════════════════════════════════════════

    def _init_episode_buffer(self):
        buf = {
            'mineral_maps': [[] for _ in range(self.config.num_robots)],
            'positions':    [[] for _ in range(self.config.num_robots)],
            'actions':      [[] for _ in range(self.config.num_robots)],
            'rewards':      [[] for _ in range(self.config.num_robots)],
            'rewards_curiosity': [[] for _ in range(self.config.num_robots)],
            'global_states': [],
            'dones': [],
            'timestamps': []
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

            # Mise à jour du timestamp pour le watchdog
            self._last_robot_time[robot_id] = time.time()

            # Observation locale (décodage base64)
            self.current_episode['mineral_maps'][robot_id].append(
                self._b64_to_arr(
                    step_data['mineral_map'], (obs_size, obs_size, n_ch)
                ).transpose(2, 0, 1)
            )
            self.current_episode['positions'][robot_id].append(
                step_data['position']
            )
            self.current_episode['actions'][robot_id].append(
                step_data['action']
            )

            # Reward = extrinsic + curiosity
            r_ext = step_data['reward']
            r_cur = step_data.get('reward_curiosity', 0.0)
            self.current_episode['rewards'][robot_id].append(r_ext + r_cur)
            self.current_episode['rewards_curiosity'][robot_id].append(r_cur)

            # Multi-scale : observation régionale (décodage base64)
            if (self.config.use_multi_scale and
                    'regional_map' in step_data and
                    'regional_maps' in self.current_episode):
                self.current_episode['regional_maps'][robot_id].append(
                    self._b64_to_arr(
                        step_data['regional_map'], (obs_size, obs_size, n_ch)
                    ).transpose(2, 0, 1)
                )

            # Synchronisation des robots
            if all(len(lst) > self.global_step
                   for lst in self.current_episode['actions']):
                gs_str = step_data.get('global_state', '')
                if gs_str:
                    global_state = self._b64_to_arr(
                        gs_str, (self.config.state_shape[0],
                                 self.config.map_height,
                                 self.config.map_width)
                    )
                else:
                    global_state = np.zeros(self.config.state_shape, dtype=np.float32)

                self.current_episode['global_states'].append(global_state)
                self.current_episode['dones'].append(step_data['done'])
                self.current_episode['timestamps'].append(time.time())
                self.global_step += 1

                if step_data['done']:
                    self._end_episode()

        except Exception as e:
            self.get_logger().error(f'Error processing experience: {e}')

    def _sync_watchdog(self):
        """Détecte les robots bloqués et reset l'épisode pour débloquer."""
        if not any(self.current_episode['actions'][i]
                   for i in range(self.config.num_robots)):
            return  # Épisode vide, rien à surveiller

        now = time.time()
        for i in range(self.config.num_robots):
            if len(self.current_episode['actions'][i]) <= self.global_step:
                delay = now - self._last_robot_time.get(i, now)
                if delay > self._sync_timeout:
                    self.get_logger().warn(
                        f'Robot {i} bloqué depuis {delay:.1f}s '
                        f'(step {self.global_step}) — reset épisode'
                    )
                    self.current_episode = self._init_episode_buffer()
                    self.global_step = 0
                    self._last_robot_time = {
                        j: now for j in range(self.config.num_robots)
                    }
                    return

    def _end_episode(self):
        try:
            episode_data = {
                'mineral_maps': [
                    np.stack(maps)
                    for maps in self.current_episode['mineral_maps']
                ],
                'positions': [
                    np.stack(pos)
                    for pos in self.current_episode['positions']
                ],
                'actions': [
                    np.array(act)
                    for act in self.current_episode['actions']
                ],
                'rewards': [
                    np.array(rew)
                    for rew in self.current_episode['rewards']
                ],
                'global_states': np.stack(
                    self.current_episode['global_states']
                ),
                'dones': np.array(self.current_episode['dones'])
            }

            # Multi-scale
            if (self.config.use_multi_scale and
                    'regional_maps' in self.current_episode and
                    all(len(r) > 0
                        for r in self.current_episode['regional_maps'])):
                episode_data['regional_maps'] = [
                    np.stack(reg)
                    for reg in self.current_episode['regional_maps']
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
                self.current_episode = self._init_episode_buffer()
                self.global_step = 0
                return

            # === MODE ENTRAÎNEMENT ===
            self.replay_buffer.add_episode(episode_data)
            self.episode_count += 1

            self._log_episode_csv(episode_data, total_reward)

            self._reward_window.append(total_reward)
            ma10 = float(np.mean(self._reward_window))

            robot_rewards = [float(np.sum(rew)) for rew in episode_data['rewards']]
            minerals_per_robot = [
                int(np.sum(rew > 50)) for rew in episode_data['rewards']
            ]
            total_minerals = sum(minerals_per_robot)

            # TensorBoard — métriques générales
            ep = self.episode_count
            self.writer.add_scalar('Episode/TotalReward', total_reward, ep)
            self.writer.add_scalar('Episode/TotalReward_MA10', ma10, ep)
            self.writer.add_scalar(
                'Episode/AvgReward', total_reward / max(steps, 1), ep
            )
            self.writer.add_scalar('Episode/Steps', steps, ep)
            self.writer.add_scalar('Episode/Epsilon', self.epsilon, ep)
            self.writer.add_scalar('Episode/BufferSize', len(self.replay_buffer), ep)
            self.writer.add_scalar('Episode/MineralsDetected', total_minerals, ep)
            for i, (rr, mn) in enumerate(zip(robot_rewards, minerals_per_robot)):
                self.writer.add_scalar(f'Robots/Robot{i}_Reward', rr, ep)
                self.writer.add_scalar(f'Robots/Robot{i}_Minerals', mn, ep)

            # ── TensorBoard — CONTRIBUTION 1 : Geo-ICM ────────────────
            if self.config.use_icm:
                curiosity_per_robot = [
                    float(np.sum(self.current_episode['rewards_curiosity'][i]))
                    for i in range(self.config.num_robots)
                ]
                total_curiosity = sum(curiosity_per_robot)
                avg_curiosity = total_curiosity / max(steps * self.config.num_robots, 1)
                self.writer.add_scalar('ICM/TotalCuriosity', total_curiosity, ep)
                self.writer.add_scalar('ICM/AvgCuriosityPerStep', avg_curiosity, ep)
                for i, cr in enumerate(curiosity_per_robot):
                    self.writer.add_scalar(f'ICM/Robot{i}_Curiosity', cr, ep)

            # ── TensorBoard — CONTRIBUTION 3 : Communication ──────────
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

            self.get_logger().info(
                f'Episode {self.episode_count} - '
                f'Steps: {steps}, Reward: {total_reward:.1f}, '
                f'Minerals: {total_minerals} {minerals_per_robot}'
            )

            if self.episode_count % self.eval_freq == 0:
                self._start_eval_mode()

            # Signaler au serveur de régénérer la carte minérale
            reset_msg = String()
            reset_msg.data = json.dumps({
                'episode': self.episode_count,
                'timestamp': time.time()
            })
            self.episode_reset_pub.publish(reset_msg)

            self.current_episode = self._init_episode_buffer()
            self.global_step = 0

        except Exception as e:
            self.get_logger().error(f'Error ending episode: {e}')

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

        self.writer.add_scalar('Eval/AvgReward', avg_reward, self.eval_round)
        self.writer.add_scalar('Eval/AvgRewardPerStep', avg_rps, self.eval_round)
        self.writer.add_scalar('Eval/AvgSteps', avg_steps, self.eval_round)

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
        """Effectue un pas d'entraînement QMIX + ICM."""
        if len(self.replay_buffer) < self.config.batch_size:
            return

        try:
            batch = self.replay_buffer.sample_batch()

            # === 1. Loss QMIX (avec multi-scale + comm) ===
            loss, q_tot_mean, q_tot_std = self._compute_loss(batch)

            self.optimizer.zero_grad()
            loss.backward()

            grad_norm = 0.0
            if self.config.grad_clip > 0:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(
                    self.qmix_network.parameters(),
                    self.config.grad_clip
                ))
            self.optimizer.step()

            # === 2. Loss ICM (Contribution 1) ===
            icm_loss = 0.0
            if self.config.use_icm and self.icm is not None:
                icm_loss = self._train_icm(batch)

            self.train_step += 1

            # Target network update
            if self.train_step % self.config.target_update_freq == 0:
                self.target_network.update(self.qmix_network)
                self.get_logger().info(
                    f'Target network updated at step {self.train_step}'
                )

            # Logging
            self._log_training_csv(loss.item(), icm_loss)

            ts = self.train_step
            self.writer.add_scalar('Train/Loss', loss.item(), ts)
            self.writer.add_scalar('Train/Epsilon', self.epsilon, ts)
            self.writer.add_scalar('Train/GradNorm', grad_norm, ts)
            self.writer.add_scalar('Train/QTot_Mean', q_tot_mean, ts)
            self.writer.add_scalar('Train/QTot_Std', q_tot_std, ts)

            if self.config.use_icm:
                self.writer.add_scalar('Train/ICM_Loss', icm_loss, ts)

            if self.train_step % 10 == 0:
                self.get_logger().info(
                    f'Train step {self.train_step} - '
                    f'Loss: {loss.item():.4f}, '
                    f'ICM: {icm_loss:.4f}, '
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

    def _train_icm(self, batch):
        """
        Entraîne le forward model Geo-ICM.

        Pour chaque timestep t dans le batch :
          1. Extraire features φ(s_t) et φ(s_{t+1}) via le CNN (detached)
          2. L_ICM = ||forward_model(φ(s_t), a_t) - φ(s_{t+1})||²
        """
        T = batch['global_states'].size(0)
        B = batch['global_states'].size(1)

        total_icm_loss = 0.0
        valid_steps = 0

        regional_maps = batch.get('regional_maps', None)

        for t in range(T - 1):
            # Vérifier que toutes les actions sont valides
            actions_t = [act[t] for act in batch['actions']]
            valid = torch.stack([a >= 0 for a in actions_t], dim=1).all(dim=1)
            if not valid.any():
                continue

            # Extraire features CNN pour tous les agents au pas t et t+1
            with torch.no_grad():
                maps_t = [maps[t] for maps in batch['mineral_maps']]
                maps_t1 = [maps[t+1] for maps in batch['mineral_maps']]

                regs_t = None
                regs_t1 = None
                if regional_maps is not None:
                    regs_t = [reg[t] for reg in regional_maps]
                    regs_t1 = [reg[t+1] for reg in regional_maps]

                features_t = self.qmix_network.extract_all_features(maps_t, regs_t)
                features_t1 = self.qmix_network.extract_all_features(maps_t1, regs_t1)

            # Loss ICM pour chaque agent (réseau partagé → même forward model)
            for i in range(self.config.num_robots):
                actions_clamped = actions_t[i].clamp(min=0)
                agent_loss = self.icm.compute_loss(
                    features_t[i].detach(),
                    actions_clamped,
                    features_t1[i].detach()
                )
                total_icm_loss += agent_loss
                valid_steps += 1

        if valid_steps == 0:
            return 0.0

        avg_icm_loss = total_icm_loss / valid_steps
        self.icm_optimizer.zero_grad()
        avg_icm_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.icm.parameters(), 5.0)
        self.icm_optimizer.step()

        return float(avg_icm_loss.item())

    def _compute_loss(self, batch):
        """
        Loss TD(n) QMIX — forward pass VECTORISÉ sur tous les timesteps.

        Avant : T-1 forward passes séquentiels (boucle for t in range(T-1))
        Après : 1 forward pass online + 1 forward pass target sur (T*B) éléments
        Gain  : ~T× plus rapide (T ≈ 300 timesteps par épisode)
        """
        T = batch['global_states'].size(0)
        B = batch['global_states'].size(1)
        TB = T * B

        regional_maps = batch.get('regional_maps', None)

        # ── 1. Aplatir (T,B,...) → (T*B,...) pour forward pass unique ────
        maps_flat = [maps.reshape(TB, *maps.shape[2:])
                     for maps in batch['mineral_maps']]
        pos_flat  = [pos.reshape(TB, *pos.shape[2:])
                     for pos in batch['positions']]
        gs_flat   = batch['global_states'].reshape(
            TB, *batch['global_states'].shape[2:]
        )
        regs_flat = ([reg.reshape(TB, *reg.shape[2:]) for reg in regional_maps]
                     if regional_maps is not None else None)

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
        actions = batch['actions']  # list[N] of (T, B)
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
        rewards = torch.stack(batch['rewards'], dim=2).sum(dim=2)  # (T, B)
        dones   = batch['dones'].float()                            # (T, B)
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

            # ICM weights
            if self.config.use_icm and self.icm is not None:
                data['icm_state_dict'] = self.icm.state_dict()

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
        if self.is_eval_mode:
            msg = Float32()
            msg.data = 0.0
            self.epsilon_pub.publish(msg)
            return

        self.epsilon = max(
            self.config.epsilon_end,
            self.config.epsilon_start -
            (self.config.epsilon_start - self.config.epsilon_end) *
            min(1.0, self.train_step / self.config.epsilon_decay)
        )

        msg = Float32()
        msg.data = float(self.epsilon)
        self.epsilon_pub.publish(msg)

    def save_checkpoint(self):
        """Sauvegarde un checkpoint (QMIX + ICM)."""
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

            if self.config.use_icm and self.icm is not None:
                data['icm_state_dict'] = self.icm.state_dict()
                data['icm_optimizer_state_dict'] = (
                    self.icm_optimizer.state_dict()
                )

            torch.save(data, path)
            torch.save(data, latest_path)

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

        except Exception as e:
            self.get_logger().error(f'Error saving checkpoint: {e}')


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    node = QMIXTrainerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down QMIX Trainer...')
    finally:
        node.get_logger().info('Sauvegarde finale du checkpoint...')
        node.save_checkpoint()
        node.writer.flush()
        node.writer.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
