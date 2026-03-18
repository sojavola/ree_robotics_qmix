#!/usr/bin/env python3

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
import time
import csv
from collections import deque
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from .networks import QMixNetwork, QMixTargetNetwork
from .replay_buffer import QMIXReplayBuffer
from .config import QMIXConfig

class QMIXTrainerNode(Node):
    """Nœud d'entraînement centralisé QMIX"""
    
    def __init__(self):
        super().__init__('qmix_trainer')
        
        # Charger la configuration
        self.declare_parameter('config_file', '')
        config_file = self.get_parameter('config_file').get_parameter_value().string_value
        
        if config_file:
            self.config = QMIXConfig.from_yaml(config_file)
        else:
            self.config = QMIXConfig()
        
        self.get_logger().info('🚀 Initializing QMIX Trainer Node')
        self.get_logger().info(f'📊 Config: {self.config}')
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'🤖 Using device: {self.device}')
        
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
        
        # Compteurs
        self.train_step = 0
        self.episode_count = 0
        self.global_step = 0
        self.epsilon = self.config.epsilon_start

        # Fenêtre glissante pour les moving averages (TensorBoard)
        self._reward_window = deque(maxlen=10)   # lissage sur 10 épisodes

        # === EVAL SPLIT ===
        self.eval_freq = 20          # Évaluation tous les N épisodes d'entraînement
        self.eval_episodes = 1       # Nombre d'épisodes d'évaluation par round
        self.is_eval_mode = False
        self.eval_episode_count = 0
        self.eval_round = 0
        self.eval_rewards_accum = []
        self.eval_steps_accum = []
        
        # Publishers
        self.weight_update_pub = self.create_publisher(
            String, 
            self.config.trainer_update_topic, 
            10
        )
        self.epsilon_pub = self.create_publisher(
            Float32,
            self.config.trainer_epsilon_topic,
            10
        )
        
        # Subscribers pour les expériences des agents
        self.exp_sub = self.create_subscription(
            String,
            self.config.agent_exp_topic,
            self.experience_callback,
            10
        )
        
        # Timers
        self.train_timer = self.create_timer(1.0, self.train_step_callback)
        self.epsilon_timer = self.create_timer(5.0, self.publish_epsilon)
        self.save_timer = self.create_timer(60.0, self.save_checkpoint)
        
        # Dossiers de sauvegarde — chemin absolu dans le home
        self.save_dir = os.path.expanduser('~/.qmix/models')
        os.makedirs(self.save_dir, exist_ok=True)

        # CSV logging
        self.log_dir = os.path.expanduser('~/.qmix/logs')
        os.makedirs(self.log_dir, exist_ok=True)
        self._init_csv_loggers()

        # TensorBoard
        tb_dir = os.path.expanduser('~/.qmix/tensorboard')
        os.makedirs(tb_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=tb_dir)
        self.get_logger().info(f'📈 TensorBoard → {tb_dir}')
        self.get_logger().info(f'   Lance avec : tensorboard --logdir {tb_dir}')

        # Continuous learning : charger le dernier checkpoint si disponible
        self._load_checkpoint_if_exists()

        self.get_logger().info('✅ QMIX Trainer Node initialized')
    
    def _init_csv_loggers(self):
        """Crée les fichiers CSV de logs (append si déjà existants)"""
        self._episodes_csv = os.path.join(self.log_dir, 'episodes.csv')
        self._training_csv = os.path.join(self.log_dir, 'training.csv')

        # episodes.csv : une ligne par épisode
        if not os.path.exists(self._episodes_csv):
            with open(self._episodes_csv, 'w', newline='') as f:
                csv.writer(f).writerow([
                    'episode', 'steps', 'total_reward',
                    'avg_reward_per_step', 'epsilon',
                    'buffer_size', 'timestamp'
                ])

        # training.csv : une ligne par train step
        if not os.path.exists(self._training_csv):
            with open(self._training_csv, 'w', newline='') as f:
                csv.writer(f).writerow([
                    'train_step', 'loss', 'epsilon', 'timestamp'
                ])

        # eval.csv : une ligne par round d'évaluation
        self._eval_csv = os.path.join(self.log_dir, 'eval.csv')
        if not os.path.exists(self._eval_csv):
            with open(self._eval_csv, 'w', newline='') as f:
                csv.writer(f).writerow([
                    'eval_round', 'avg_reward', 'avg_reward_per_step',
                    'avg_steps', 'episode_count', 'train_step', 'timestamp'
                ])

        self.get_logger().info(f'📁 CSV logs → {self.log_dir}')

    def _log_episode_csv(self, episode_data, total_reward):
        """Écrit une ligne dans episodes.csv"""
        try:
            steps = len(episode_data['dones'])
            with open(self._episodes_csv, 'a', newline='') as f:
                csv.writer(f).writerow([
                    self.episode_count,
                    steps,
                    round(total_reward, 3),
                    round(total_reward / max(steps, 1), 3),
                    round(self.epsilon, 4),
                    len(self.replay_buffer),
                    datetime.now().isoformat()
                ])
        except Exception as e:
            self.get_logger().error(f'❌ CSV episode log error: {e}')

    def _log_training_csv(self, loss):
        """Écrit une ligne dans training.csv"""
        try:
            with open(self._training_csv, 'a', newline='') as f:
                csv.writer(f).writerow([
                    self.train_step,
                    round(float(loss), 6),
                    round(self.epsilon, 4),
                    datetime.now().isoformat()
                ])
        except Exception as e:
            self.get_logger().error(f'❌ CSV training log error: {e}')

    def _init_networks(self):
        """Initialise les réseaux QMIX"""
        state_shape = self.config.state_shape
        
        # Réseau principal
        self.qmix_network = QMixNetwork(
            state_shape=state_shape,
            num_agents=self.config.num_robots,
            num_actions=self.config.num_actions,
            hidden_dim=self.config.hidden_dim,
            hyper_dim=self.config.hyper_dim,
            local_obs_size=self.config.local_obs_size
        ).to(self.device)
        
        # Réseau cible
        self.target_network = QMixTargetNetwork(self.qmix_network).to(self.device)
        self.target_network.update(self.qmix_network)
        
        # Optimiseur
        self.optimizer = optim.Adam(
            self.qmix_network.parameters(),
            lr=self.config.learning_rate
        )
        
        self.get_logger().info('🧠 Réseaux QMIX initialisés')

    def _load_checkpoint_if_exists(self):
        """Charge le dernier checkpoint pour le continuous learning"""
        latest_path = os.path.join(os.path.expanduser('~/.qmix/models'), 'latest.pt')
        if not os.path.exists(latest_path):
            self.get_logger().info('📂 Aucun checkpoint trouvé — démarrage à zéro')
            return
        try:
            checkpoint = torch.load(latest_path, map_location=self.device)
            self.qmix_network.load_state_dict(checkpoint['qmix_state_dict'])
            self.target_network.update(self.qmix_network)
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.train_step = checkpoint.get('train_step', 0)
            self.episode_count = checkpoint.get('episode_count', 0)
            self.global_step = checkpoint.get('global_step', 0)
            self.epsilon = checkpoint.get('epsilon', self.config.epsilon_start)
            self.eval_round = checkpoint.get('eval_round', 0)
            self.get_logger().info(
                f'✅ Checkpoint chargé — train_step={self.train_step}, '
                f'episodes={self.episode_count}, '
                f'global_step={self.global_step}, '
                f'epsilon={self.epsilon:.3f}'
            )
        except Exception as e:
            self.get_logger().error(f'❌ Erreur chargement checkpoint: {e} — démarrage à zéro')

    def _init_episode_buffer(self):
        """Initialise un buffer pour l'épisode en cours"""
        return {
            'mineral_maps': [[] for _ in range(self.config.num_robots)],
            'positions': [[] for _ in range(self.config.num_robots)],
            'actions': [[] for _ in range(self.config.num_robots)],
            'rewards': [[] for _ in range(self.config.num_robots)],
            'global_states': [],
            'dones': [],
            'timestamps': []
        }
    
    def experience_callback(self, msg: String):
        """Callback pour recevoir les expériences des agents"""
        try:
            data = json.loads(msg.data)
            
            robot_id = data['robot_id']
            step_data = data['step_data']
            
            # Ajouter au buffer d'épisode
            obs_size = self.config.local_obs_size
            n_ch = self.config.state_shape[0]
            # Agent envoie (H, W, C) → reshape en (H,W,C) puis transpose en (C,H,W)
            self.current_episode['mineral_maps'][robot_id].append(
                np.array(step_data['mineral_map'])
                .reshape(obs_size, obs_size, n_ch)
                .transpose(2, 0, 1)  # (C, H, W)
            )
            self.current_episode['positions'][robot_id].append(
                step_data['position']
            )
            self.current_episode['actions'][robot_id].append(
                step_data['action']
            )
            self.current_episode['rewards'][robot_id].append(
                step_data['reward']
            )
            
            # Vérifier si c'est le dernier robot à envoyer pour ce step
            # (implémentation simple: on suppose que tous les robots envoient en même temps)
            if all(len(lst) > self.global_step for lst in self.current_episode['actions']):
                # Ajouter l'état global
                # global_state est déjà (4, H, W) sérialisé par l'agent
                self.current_episode['global_states'].append(
                    np.array(step_data['global_state']).reshape(
                        self.config.state_shape[0],
                        self.config.map_height,
                        self.config.map_width
                    )
                )
                self.current_episode['dones'].append(
                    step_data['done']
                )
                self.current_episode['timestamps'].append(time.time())
                
                self.global_step += 1
                
                # Vérifier si l'épisode est terminé
                if step_data['done']:
                    self._end_episode()
                    
        except Exception as e:
            self.get_logger().error(f'❌ Error processing experience: {e}')
    
    def _end_episode(self):
        """Termine l'épisode et ajoute au replay buffer (ou évalue si eval_mode)"""
        try:
            # Convertir en arrays numpy
            episode_data = {
                'mineral_maps': [
                    np.stack(maps) for maps in self.current_episode['mineral_maps']
                ],
                'positions': [
                    np.stack(pos) for pos in self.current_episode['positions']
                ],
                'actions': [
                    np.array(act) for act in self.current_episode['actions']
                ],
                'rewards': [
                    np.array(rew) for rew in self.current_episode['rewards']
                ],
                'global_states': np.stack(self.current_episode['global_states']),
                'dones': np.array(self.current_episode['dones'])
            }

            steps = len(episode_data['dones'])
            total_reward = sum(sum(rew) for rew in episode_data['rewards'])

            # === MODE ÉVALUATION ===
            if self.is_eval_mode:
                self.eval_rewards_accum.append(total_reward)
                self.eval_steps_accum.append(steps)
                self.eval_episode_count += 1
                self.get_logger().info(
                    f'🔬 Eval épisode {self.eval_episode_count}/{self.eval_episodes} - '
                    f'Reward: {total_reward:.1f}, Steps: {steps}'
                )
                if self.eval_episode_count >= self.eval_episodes:
                    self._end_eval_mode()
                # Ne pas ajouter au replay buffer pendant l'éval
                self.current_episode = self._init_episode_buffer()
                self.global_step = 0
                return

            # === MODE ENTRAÎNEMENT ===
            self.replay_buffer.add_episode(episode_data)
            self.episode_count += 1

            self._log_episode_csv(episode_data, total_reward)

            # Moving average
            self._reward_window.append(total_reward)
            ma10 = float(np.mean(self._reward_window))

            # Récompense par robot
            robot_rewards = [float(np.sum(rew)) for rew in episode_data['rewards']]

            # Minéraux détectés : reward > 50 = détection confirmée
            minerals_per_robot = [int(np.sum(rew > 50)) for rew in episode_data['rewards']]
            total_minerals = sum(minerals_per_robot)

            # TensorBoard — métriques par épisode
            ep = self.episode_count
            self.writer.add_scalar('Episode/TotalReward',      total_reward,                 ep)
            self.writer.add_scalar('Episode/TotalReward_MA10', ma10,                         ep)
            self.writer.add_scalar('Episode/AvgReward',        total_reward / max(steps, 1), ep)
            self.writer.add_scalar('Episode/Steps',            steps,                        ep)
            self.writer.add_scalar('Episode/Epsilon',          self.epsilon,                 ep)
            self.writer.add_scalar('Episode/BufferSize',       len(self.replay_buffer),      ep)
            self.writer.add_scalar('Episode/MineralsDetected', total_minerals,               ep)
            for i, (rr, mn) in enumerate(zip(robot_rewards, minerals_per_robot)):
                self.writer.add_scalar(f'Robots/Robot{i}_Reward',   rr, ep)
                self.writer.add_scalar(f'Robots/Robot{i}_Minerals', mn, ep)

            self.get_logger().info(
                f'📊 Épisode {self.episode_count} terminé - '
                f'Steps: {steps}, Reward: {total_reward:.1f}, '
                f'💎 Minéraux: {total_minerals} {minerals_per_robot}'
            )

            # Déclencher l'évaluation tous les eval_freq épisodes
            if self.episode_count % self.eval_freq == 0:
                self._start_eval_mode()

            # Réinitialiser le buffer d'épisode
            self.current_episode = self._init_episode_buffer()
            self.global_step = 0

        except Exception as e:
            self.get_logger().error(f'❌ Error ending episode: {e}')

    def _start_eval_mode(self):
        """Passe en mode évaluation (epsilon=0, pas d'ajout au buffer)"""
        self.is_eval_mode = True
        self.eval_episode_count = 0
        self.eval_rewards_accum = []
        self.eval_steps_accum = []
        self.eval_round += 1
        # Publier epsilon=0 immédiatement pour les agents
        msg = Float32()
        msg.data = 0.0
        self.epsilon_pub.publish(msg)
        self.get_logger().info(
            f'🔬 === ÉVALUATION #{self.eval_round} démarrée '
            f'({self.eval_episodes} épisodes, ε=0.0) ==='
        )

    def _end_eval_mode(self):
        """Termine le round d'évaluation et log les résultats"""
        avg_reward = float(np.mean(self.eval_rewards_accum))
        avg_steps = float(np.mean(self.eval_steps_accum))
        avg_rps = float(np.mean([
            r / max(s, 1)
            for r, s in zip(self.eval_rewards_accum, self.eval_steps_accum)
        ]))

        # CSV
        try:
            with open(self._eval_csv, 'a', newline='') as f:
                csv.writer(f).writerow([
                    self.eval_round,
                    round(avg_reward, 3),
                    round(avg_rps, 4),
                    round(avg_steps, 1),
                    self.episode_count,
                    self.train_step,
                    datetime.now().isoformat()
                ])
        except Exception as e:
            self.get_logger().error(f'❌ CSV eval log error: {e}')

        # TensorBoard
        self.writer.add_scalar('Eval/AvgReward',        avg_reward, self.eval_round)
        self.writer.add_scalar('Eval/AvgRewardPerStep', avg_rps,    self.eval_round)
        self.writer.add_scalar('Eval/AvgSteps',         avg_steps,  self.eval_round)

        self.get_logger().info(
            f'🔬 === ÉVALUATION #{self.eval_round} terminée === '
            f'Reward moy: {avg_reward:.1f}, '
            f'Reward/step: {avg_rps:.3f}, '
            f'Steps moy: {avg_steps:.1f}'
        )

        self.is_eval_mode = False
    
    def train_step_callback(self):
        """Effectue un pas d'entraînement"""
        if len(self.replay_buffer) < self.config.batch_size:
            return
        
        try:
            # Échantillonner un batch
            batch = self.replay_buffer.sample_batch()
            
            # Calculer la perte
            loss, q_tot_mean, q_tot_std = self._compute_loss(batch)

            # Optimisation
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping — clip_grad_norm_ retourne la norme totale avant clipping
            grad_norm = 0.0
            if self.config.grad_clip > 0:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(
                    self.qmix_network.parameters(),
                    self.config.grad_clip
                ))

            self.optimizer.step()
            self.train_step += 1

            # Mettre à jour le réseau cible
            if self.train_step % self.config.target_update_freq == 0:
                self.target_network.update(self.qmix_network)
                self.get_logger().info(f'🎯 Target network updated at step {self.train_step}')

            # CSV logging à chaque step
            self._log_training_csv(loss.item())

            # TensorBoard — métriques par train step
            ts = self.train_step
            self.writer.add_scalar('Train/Loss',        loss.item(),  ts)
            self.writer.add_scalar('Train/Epsilon',     self.epsilon, ts)
            self.writer.add_scalar('Train/GradNorm',    grad_norm,    ts)
            self.writer.add_scalar('Train/QTot_Mean',   q_tot_mean,   ts)
            self.writer.add_scalar('Train/QTot_Std',    q_tot_std,    ts)

            # Console logging tous les 10 steps
            if self.train_step % 10 == 0:
                self.get_logger().info(
                    f'📚 Train step {self.train_step} - '
                    f'Loss: {loss.item():.4f}, '
                    f'QTot: {q_tot_mean:.1f}±{q_tot_std:.1f}, '
                    f'GradNorm: {grad_norm:.2f}, '
                    f'ε: {self.epsilon:.3f}'
                )
            
            # Histogrammes des poids CNN + feature maps (tous les 50 steps)
            if self.train_step % 50 == 0:
                self._publish_network_weights()
                self._log_cnn_stats(batch, ts)
                
        except Exception as e:
            self.get_logger().error(f'❌ Training error: {e}')
            import traceback
            traceback.print_exc()
    
    def _log_cnn_stats(self, batch, step):
        """
        Histogrammes des poids CNN + feature maps visuelles dans TensorBoard.
        Appelé tous les 50 train steps — pas de surcharge mémoire.
        """
        encoder = self.qmix_network.shared_agent_network.encoder

        # ── 1. Histogrammes des poids (distribution) ─────────────────────
        with torch.no_grad():
            self.writer.add_histogram('CNN/conv1_weights', encoder.conv1.weight, step)
            self.writer.add_histogram('CNN/conv2_weights', encoder.conv2.weight, step)
            self.writer.add_histogram('CNN/conv3_weights', encoder.conv3.weight, step)
            self.writer.add_histogram('CNN/fc_weights',    encoder.fc.weight,    step)

        # ── 2. Feature maps visuelles (conv1 seulement — 32 filtres) ─────
        # On prend le 1er sample du batch (robot 0, timestep 0)
        try:
            with torch.no_grad():
                sample_map = batch['mineral_maps'][0][0].unsqueeze(0)  # (1, C, H, W)

                activations = {}
                def make_hook(name):
                    def hook(module, input, output):
                        activations[name] = output.detach().cpu()
                    return hook

                hooks = [
                    encoder.conv1.register_forward_hook(make_hook('conv1')),
                    encoder.conv2.register_forward_hook(make_hook('conv2')),
                ]
                encoder(sample_map)
                for h in hooks:
                    h.remove()

                for name, acts in activations.items():
                    # acts: (1, C_out, H, W) → (C_out, 1, H, W)
                    feat = acts[0].unsqueeze(1)  # (C_out, 1, H, W)
                    # Normaliser chaque filtre dans [0,1] pour la visualisation
                    f_min = feat.flatten(1).min(1)[0].view(-1, 1, 1, 1)
                    f_max = feat.flatten(1).max(1)[0].view(-1, 1, 1, 1)
                    denom = (f_max - f_min).clamp(min=1e-6)
                    feat = (feat - f_min) / denom
                    self.writer.add_images(f'CNN/{name}_feature_maps', feat, step)

                # ── 3. Input brut (6 canaux → 3 premiers + 3 suivants) ───
                inp = sample_map[0].cpu()  # (6, H, W)
                # Canaux minéraux 0-3 : prendre le max comme image de densité
                mineral_density = inp[:4].max(0)[0].unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
                f_min = mineral_density.min(); f_max = mineral_density.max()
                if f_max > f_min:
                    mineral_density = (mineral_density - f_min) / (f_max - f_min)
                self.writer.add_images('CNN/input_mineral_density', mineral_density, step)

        except Exception as e:
            self.get_logger().debug(f'CNN activation log skipped: {e}')

    def _compute_nstep_target(self, batch, t, T, B):
        """
        Calcule la cible TD(n) pour le timestep t :
            G_t^n = Σ_{k=0}^{n-1} γ^k * r_{t+k} + γ^n * Q_target(s_{t+n})
        Si l'épisode se termine avant n pas, on cesse d'accumuler et de bootstrapper.
        """
        n = self.config.n_steps
        nstep_return = torch.zeros(B, device=self.device)
        discount = 1.0
        # not_done[b] = 1 si l'épisode b est encore actif
        not_done = torch.ones(B, device=self.device)

        for k in range(n):
            tk = t + k
            if tk >= T - 1:
                break  # plus de timesteps disponibles

            # Somme des rewards de tous les agents au pas t+k
            r_tk = torch.stack(
                [rew[tk] for rew in batch['rewards']], dim=1
            ).sum(dim=1)  # (B,)

            nstep_return = nstep_return + discount * r_tk * not_done

            # Mettre à jour le masque : épisode terminé = plus de reward futur
            not_done = not_done * (1.0 - batch['dones'][tk].float())
            discount *= self.config.gamma

        # Bootstrap depuis Q_target au pas t+n (si pas terminé et dans les bornes)
        bootstrap_t = t + n
        if bootstrap_t < T:
            with torch.no_grad():
                gs_tn = batch['global_states'][bootstrap_t]
                _, next_q_ind = self.target_network.qmix_network(
                    [maps[bootstrap_t] for maps in batch['mineral_maps']],
                    [pos[bootstrap_t]  for pos  in batch['positions']],
                    gs_tn
                )
                next_q_max = torch.stack(
                    [q.max(1)[0] for q in next_q_ind], dim=1
                )  # (B, num_robots)
                q_tot_tn = self.target_network.qmix_network.mixing_network(
                    next_q_max, gs_tn.view(B, -1)
                )  # (B,)
            nstep_return = nstep_return + discount * q_tot_tn * not_done

        return nstep_return

    def _compute_loss(self, batch):
        """
        Calcule la perte TD(n)-error QMIX :
            L = ( Q_tot(s,a) - G_t^n )²
        G_t^n = Σ_{k=0}^{n-1} γ^k*r_{t+k} + γ^n*Q_target(s_{t+n})
        Les timesteps paddés (action == -1) sont masqués et ignorés.
        """
        T = batch['global_states'].size(0)
        B = batch['global_states'].size(1)

        total_loss = torch.tensor(0.0, device=self.device)
        valid_steps = 0
        q_tot_mean = 0.0
        q_tot_std  = 0.0

        for t in range(T - 1):
            actions_t = [act[t] for act in batch['actions']]

            # Masque : TOUS les robots ont une action valide (≥ 0)
            valid = torch.stack([a >= 0 for a in actions_t], dim=1).all(dim=1)  # (B,)
            if not valid.any():
                continue

            # Clamp pour éviter crash du gather sur -1 (padding)
            actions_clamped = [a.clamp(min=0) for a in actions_t]

            mineral_maps_t = [maps[t] for maps in batch['mineral_maps']]
            positions_t    = [pos[t]  for pos  in batch['positions']]
            global_state_t = batch['global_states'][t]

            # ── Q-values online (toutes les actions) ──────────────────────
            _, q_individuals_t = self.qmix_network(
                mineral_maps_t, positions_t, global_state_t
            )

            # Q-values des actions CHOISIES → (B, num_robots)
            chosen_qs = torch.stack([
                q_individuals_t[i].gather(1, actions_clamped[i].unsqueeze(1)).squeeze(1)
                for i in range(self.config.num_robots)
            ], dim=1)

            # Q_tot online sur les actions choisies
            q_tot_t = self.qmix_network.mixing_network(
                chosen_qs, global_state_t.view(B, -1)
            )  # (B,)

            # ── Cible TD(n) ───────────────────────────────────────────────
            targets = self._compute_nstep_target(batch, t, T, B)

            # ── TD-error masqué (ignorer les timesteps paddés) ────────────
            # Huber loss (smooth_l1) : quadratique pour petites erreurs, linéaire pour grandes
            # → robuste aux grandes valeurs de Q_tot (évite l'explosion de loss MSE)
            td_error = F.smooth_l1_loss(q_tot_t, targets.detach(), reduction='none')
            loss = (td_error * valid.float()).sum() / valid.float().sum()
            total_loss += loss
            valid_steps += 1

            # Stats Q_tot pour monitoring (détachées du graphe)
            with torch.no_grad():
                q_vals = q_tot_t[valid]
                if q_vals.numel() > 0:
                    q_tot_mean += float(q_vals.mean())
                    q_tot_std  += float(q_vals.std() if q_vals.numel() > 1 else 0.0)

        if valid_steps == 0:
            return torch.tensor(0.0, requires_grad=True, device=self.device), 0.0, 0.0

        return total_loss / valid_steps, float(q_tot_mean / valid_steps), float(q_tot_std / valid_steps)
    
    def _publish_network_weights(self):
        """Publie les poids du réseau pour les agents"""
        try:
            # Sauvegarder temporairement
            save_path = '/tmp/qmix_weights.pt'
            torch.save({
                'state_dict': self.qmix_network.state_dict(),
                'train_step': self.train_step,
                'episode_count': self.episode_count
            }, save_path)
            
            # Publier le chemin
            msg = String()
            msg.data = json.dumps({
                'type': 'weight_update',
                'path': save_path,
                'train_step': self.train_step,
                'timestamp': time.time()
            })
            self.weight_update_pub.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f'❌ Error publishing weights: {e}')
    
    def publish_epsilon(self):
        """Publie la valeur courante d'epsilon (0.0 en mode éval)"""
        if self.is_eval_mode:
            msg = Float32()
            msg.data = 0.0
            self.epsilon_pub.publish(msg)
            return

        # Décroissance d'epsilon (entraînement normal)
        self.epsilon = max(
            self.config.epsilon_end,
            self.config.epsilon_start -
            (self.config.epsilon_start - self.config.epsilon_end) *
            min(1.0, self.global_step / self.config.epsilon_decay)
        )

        msg = Float32()
        msg.data = float(self.epsilon)
        self.epsilon_pub.publish(msg)
    
    def save_checkpoint(self):
        """Sauvegarde un checkpoint — garde seulement les 3 derniers + latest.pt"""
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
            torch.save(data, path)
            torch.save(data, latest_path)

            # Supprimer les anciens — garder seulement les 3 derniers
            checkpoints = sorted([
                f for f in os.listdir(self.save_dir)
                if f.startswith('checkpoint_') and f.endswith('.pt')
            ])
            for old in checkpoints[:-3]:
                os.remove(os.path.join(self.save_dir, old))

            self.get_logger().info(
                f'💾 Checkpoint saved (step={self.train_step}, episodes={self.episode_count})'
            )
            
        except Exception as e:
            self.get_logger().error(f'❌ Error saving checkpoint: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = QMIXTrainerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('🛑 Shutting down QMIX Trainer...')
    finally:
        # Sauvegarde finale avant extinction (continuous learning)
        node.get_logger().info('💾 Sauvegarde finale du checkpoint...')
        node.save_checkpoint()
        node.writer.flush()
        node.writer.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()