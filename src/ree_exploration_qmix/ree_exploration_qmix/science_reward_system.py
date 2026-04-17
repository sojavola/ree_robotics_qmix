#!/usr/bin/env python3

import numpy as np
from scipy.ndimage import gaussian_filter
from collections import deque


class RealMineralRewardSystem:
    """
    Système de récompenses basé sur les minéraux réels + coverage bonus.

    Le coverage bonus remplace le Geo-ICM : récompense proportionnelle aux
    nouvelles cellules découvertes dans l'épisode courant.
    r_coverage = coverage_weight  pour chaque cellule jamais visitée cet épisode.
    """

    def __init__(self, grid_size=(100, 100), robot_id=0, coverage_weight=0.5):
        self.grid_size = grid_size
        self.robot_id = robot_id
        self.coverage_weight = coverage_weight

        self.reward_config = {
            # === RÉCOMPENSES MINÉRALES ===
            'mineral_base_reward': 50.0,
            'concentration_multiplier': 30.0,
            'high_concentration_bonus': 30.0,
            'mineral_threshold': 0.3,

            # === EXPLORATION ===
            'exploration_bonus': 1.0,       # bonus supplémentaire nouvelle cellule
            'new_zone_bonus': 2.0,          # bonus supplémentaire dans les 50 premiers steps

            # === PÉNALITÉS ===
            'step_penalty': -0.05,
            'collision_penalty': -5.0,
            'revisiting_penalty': -0.5,

            # === COVERAGE (remplace ICM) ===
            # r_coverage = coverage_weight × 1 si cellule nouvelle dans l'épisode
            # cf. __init__ coverage_weight

            # === EFFICACITÉ ===
            'efficiency_bonus': 0.3,

            # === Paramètres académiques (conservés pour heatmap legacy) ===
            'academic_penalty': -2.0,
            'clean_threshold': 0.5,
            'gaussian_sigma': 0.9,
            'gaussian_update_freq': 15,
            'discount_factor': 0.99,
        }

        # === HEATMAP (conservée pour compatibilité legacy) ===
        self.academic_heatmap = self._initialize_academic_heatmap()

        # === SUIVI DES MINÉRAUX ===
        self.minerals_collected = 0
        self.steps_without_mineral = 0
        self.last_mineral_position = None
        self.mineral_positions = []
        self.mineral_positions_set = set()
        self.concentration_history = []

        # Compteur de visites par case (decay exponentiel revisit)
        self.visit_counts: dict = {}
        # Historique 10 dernières positions (anti-loop penalty)
        self._recent_positions: deque = deque(maxlen=10)

        # === COVERAGE — cellules visitées CET épisode ===
        # Réinitialisé à chaque reset_episode()
        self._episode_visited: set = set()

        # === SUIVI EXPLORATION ===
        self.visited_positions = set()
        self.cleaned_positions = set()
        self.unique_positions_count = 0

        # === COMPTEURS ===
        self.gaussian_step = 0
        self.total_steps = 0

        # === STATISTIQUES ===
        self.total_reward = 0.0
        self.academic_reward_total = 0.0
        self.real_reward_total = 0.0
        self.episode_rewards = []

        # === COMPOSANTES REWARD — pour TensorBoard (Priorité 4) ===
        # Accumulées par épisode, remises à 0 dans reset_episode()
        self._ep_mineral: float = 0.0   # Reward/Mineral
        self._ep_coverage: float = 0.0  # Reward/Coverage
        self._ep_penalty: float = 0.0   # Reward/Penalty

    def _initialize_academic_heatmap(self):
        """Initialise la heatmap académique sparse (6 dépôts localisés)."""
        height, width = self.grid_size
        heatmap = np.zeros((height, width), dtype=np.float32)

        for _ in range(6):
            x = np.random.randint(15, width - 15)
            y = np.random.randint(15, height - 15)
            radius = np.random.randint(5, 12)

            for i in range(max(0, y - radius), min(height, y + radius)):
                for j in range(max(0, x - radius), min(width, x + radius)):
                    distance = np.sqrt((i - y)**2 + (j - x)**2)
                    if distance < radius:
                        concentration = 0.9 * (1.0 - distance / radius)
                        heatmap[i, j] = max(heatmap[i, j], concentration)

        heatmap = gaussian_filter(heatmap, sigma=1.5)
        return np.clip(heatmap, 0, 1)

    def calculate_reward(self, mineral_concentrations, position,
                         is_new_position=False, has_collision=False,
                         step_count=0, sensor_data=None):
        """
        Calcule la récompense : minérale + coverage bonus + pénalités.

        Coverage bonus = coverage_weight si la cellule est nouvelle dans cet épisode.
        Remplace le Geo-ICM qui ajoutait du bruit dans les environnements à
        carte régénérée aléatoirement.
        """
        x, y = position
        position_key = (int(x), int(y))
        self.total_steps += 1

        # Vérifier AVANT que _calculate_strategic_bonus n'ajoute à _episode_visited
        _is_new_ep = position_key not in self._episode_visited

        # === 1. RÉCOMPENSE MINÉRALE ===
        real_reward = self._calculate_real_mineral_reward(mineral_concentrations, position_key)

        # === 2. COVERAGE BONUS + PÉNALITÉS ===
        strategic_bonus = self._calculate_strategic_bonus(position_key, step_count)

        reward = real_reward + strategic_bonus

        # === 3. ACCUMULATEURS COMPOSANTES (TensorBoard Reward/*) ===
        self._ep_mineral += real_reward
        if _is_new_ep:
            self._ep_coverage += self.coverage_weight
        # Tout le reste du strategic_bonus = pénalités (step_penalty + anti-loop)
        self._ep_penalty += strategic_bonus - (self.coverage_weight if _is_new_ep else 0.0)

        # === 4. MISE À JOUR ===
        self._update_tracking(position_key, is_new_position, real_reward > 0)
        self.total_reward += reward
        self.real_reward_total += real_reward
        self.academic_reward_total = 0.0

        if np.isnan(reward) or np.isinf(reward):
            return 0.0

        return reward

    def _calculate_real_mineral_reward(self, concentrations, position_key):
        """
        Récompense pour minéraux réels avec decay exponentiel sur revisit.
        """
        if not concentrations:
            return 0.0

        max_concentration = max(concentrations) if concentrations else 0.0

        if max_concentration < self.reward_config['mineral_threshold']:
            self.steps_without_mineral += 1
            return 0.0

        self.steps_without_mineral = 0

        if position_key in self.mineral_positions_set:
            # Decay exponentiel : évite le farming sur la même case
            visit_count = self.visit_counts.get(position_key, 1)
            decay = 0.05 / (1.0 + visit_count * 0.5)
            revisit_reward = max_concentration * self.reward_config['mineral_base_reward'] * decay
            self.visit_counts[position_key] = visit_count + 1
            return revisit_reward

        # Première visite : reward plein
        mineral_reward = max_concentration * self.reward_config['mineral_base_reward']
        mineral_reward += max_concentration * self.reward_config['concentration_multiplier']

        if max_concentration > 0.7:
            mineral_reward += self.reward_config['high_concentration_bonus']

        self.minerals_collected += 1
        self.mineral_positions.append(position_key)
        self.mineral_positions_set.add(position_key)
        self.visit_counts[position_key] = 1
        self.concentration_history.append(max_concentration)
        self.last_mineral_position = position_key

        return mineral_reward

    def _calculate_strategic_bonus(self, position_key, step_count):
        """
        Bonus stratégiques :
          - Coverage bonus : reward si cellule nouvelle cet épisode (Fix C)
          - Pénalité anti-loop : pénalise les trajectoires circulaires
          - Step penalty : légère pénalité par pas (encourage l'efficacité)

        Supprimés (Fix B/C) :
          - exploration_bonus / new_zone_bonus : redondants avec coverage_weight,
            gonflaient les rewards pendant la phase d'exploration aléatoire
          - efficiency_bonus × 100 : trop agressif, distordait le signal TD
        """
        bonus = 0.0

        # === COVERAGE BONUS (seul signal d'exploration) ===
        # +coverage_weight par cellule nouvelle dans cet épisode uniquement.
        # visited_positions (global) n'est plus effacé → exploration_bonus retiré.
        if position_key not in self._episode_visited:
            bonus += self.coverage_weight

        # Marquer la cellule comme visitée cet épisode
        self._episode_visited.add(position_key)

        # === STEP PENALTY ===
        bonus += self.reward_config['step_penalty']

        # === ANTI-LOOP ===
        self._recent_positions.append(position_key)
        loop_count = list(self._recent_positions).count(position_key)
        if loop_count >= 3:
            bonus -= 0.5  # -2.0 → -0.5 : ratio 1:1 avec coverage_weight, évite reward négatif

        return bonus

    def _update_tracking(self, position_key, is_new_position, found_mineral):
        """Met à jour le suivi global des positions visitées."""
        if is_new_position and position_key not in self.visited_positions:
            self.visited_positions.add(position_key)
            self.unique_positions_count += 1

    def get_statistics(self):
        """Retourne les statistiques complètes."""
        avg_concentration = (np.mean(self.concentration_history)
                             if self.minerals_collected > 0 else 0.0)
        coverage = len(self.visited_positions) / (self.grid_size[0] * self.grid_size[1]) * 100
        total_cells = self.grid_size[0] * self.grid_size[1]
        current_priority_sum = np.sum(self.academic_heatmap)
        cleanliness_percentage = 100 * (1.0 - current_priority_sum / total_cells)

        return {
            'robot_id': self.robot_id,
            'minerals_collected': self.minerals_collected,
            'total_reward': self.total_reward,
            'academic_reward': self.academic_reward_total,
            'real_reward': self.real_reward_total,
            'visited_positions': len(self.visited_positions),
            'episode_visited': len(self._episode_visited),
            'coverage_percentage': coverage,
            'cleanliness_percentage': cleanliness_percentage,
            'avg_mineral_concentration': avg_concentration,
            'steps_without_mineral': self.steps_without_mineral,
            'gaussian_updates': self.gaussian_step // self.reward_config['gaussian_update_freq'],
            'heatmap_mean': float(np.mean(self.academic_heatmap)),
            'heatmap_std': float(np.std(self.academic_heatmap)),
            'bellman_discount': self.reward_config['discount_factor'],
        }

    def reset_episode(self):
        """
        Réinitialise pour un nouvel épisode.
        _episode_visited est remis à zéro → le coverage bonus repart à 0
        pour toutes les cellules au début de chaque épisode.

        Fix A : visited_positions N'est PAS effacé — il accumule les cellules
        visitées sur toute la vie du robot pour le suivi global (coverage %).
        L'effacement causait une inflation des rewards pendant l'exploration.
        """
        self.episode_rewards.append(self.total_reward)
        self.total_reward = 0.0
        self.academic_reward_total = 0.0
        self.real_reward_total = 0.0
        # visited_positions intentionnellement conservé (Fix A)
        self.cleaned_positions.clear()
        self.mineral_positions_set.clear()
        self.minerals_collected = 0
        self.mineral_positions = []
        self.unique_positions_count = 0
        self.steps_without_mineral = 0
        self.total_steps = 0
        self.gaussian_step = 0

        # Reset coverage et tracking
        self.visit_counts.clear()
        self._recent_positions.clear()
        self._episode_visited.clear()  # reset coverage par épisode

        # Reset composantes reward pour le prochain épisode
        self._ep_mineral  = 0.0
        self._ep_coverage = 0.0
        self._ep_penalty  = 0.0

    def get_episode_reward_components(self):
        """Retourne les composantes du reward accumulées sur l'épisode.

        Appelé par l'agent avant reset_episode() pour envoyer au trainer
        (TensorBoard Reward/Mineral, Reward/Coverage, Reward/Penalty).
        """
        return {
            'mineral':  round(self._ep_mineral,  3),
            'coverage': round(self._ep_coverage, 3),
            'penalty':  round(self._ep_penalty,  3),
        }

    def get_reward_breakdown(self, position, concentrations):
        """Retourne la décomposition détaillée des récompenses."""
        x, y = position
        position_key = (int(x), int(y))

        heatmap_value = 0.0
        if 0 <= y < self.grid_size[0] and 0 <= x < self.grid_size[1]:
            heatmap_value = self.academic_heatmap[y, x]

        max_concentration = max(concentrations) if concentrations else 0.0

        return {
            'position': position_key,
            'heatmap_value': heatmap_value,
            'max_concentration': max_concentration,
            'academic_potential': heatmap_value * self.reward_config['mineral_base_reward'],
            'real_potential': max_concentration * self.reward_config['mineral_base_reward'],
            'is_cleaned': position_key in self.cleaned_positions,
            'is_visited': position_key in self.visited_positions,
            'is_visited_this_episode': position_key in self._episode_visited,
            'gaussian_updates': self.gaussian_step // self.reward_config['gaussian_update_freq'],
        }
