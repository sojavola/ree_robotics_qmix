#!/usr/bin/env python3

import numpy as np
from scipy.ndimage import gaussian_filter


class RealMineralRewardSystem:
    """
    Système de récompenses RÉEL basé sur la logique académique
    avec heatmap, diffusion gaussienne et équation de Bellman
    """

    def __init__(self, grid_size=(100, 100), robot_id=0):
        self.grid_size = grid_size
        self.robot_id = robot_id

        # === CONFIGURATION ACADÉMIQUE OPTIMISÉE ===
        self.reward_config = {
            # === PARAMÈTRES ACADÉMIQUES (comme dans l'article) ===
            'academic_penalty': -2.0,           # Pénalité standard académique
            'clean_threshold': 0.5,             # Seuil élevé → heatmap réellement sparse
            'gaussian_sigma': 0.9,              # σ pour diffusion gaussienne
            'gaussian_update_freq': 15,         # ψ = 15 pas de temps
            'discount_factor': 0.99,            # γ pour Bellman

            # === RÉCOMPENSES MINÉRALES (calibrées) ===
            'mineral_base_reward': 50.0,        # Réduit pour éviter faux positifs
            'concentration_multiplier': 30.0,   # Multiplicateur plus réaliste
            'high_concentration_bonus': 30.0,   # Bonus pour > 0.7

            # === SEUIL MINÉRAL RÉEL ===
            'mineral_threshold': 0.3,           # Concentration minimale pour compter

            # === EXPLORATION ACADÉMIQUE ===
            'exploration_bonus': 1.0,           # Très faible comme dans l'article
            'new_zone_bonus': 2.0,              # Pour nouvelles zones

            # === PÉNALITÉS ACADÉMIQUES ===
            'step_penalty': -0.05,              # Pénalité par pas (légère)
            'collision_penalty': -5.0,          # Pour collisions
            'revisiting_penalty': -0.5,         # Pour zones revisitées

            # === BONUS STRATÉGIQUES ===
            'coverage_bonus': 0.02,             # Par % de carte exploré
            'efficiency_bonus': 0.3,            # Pour minéraux/steps
        }

        # === HEATMAP ACADÉMIQUE (simulée) ===
        self.academic_heatmap = self._initialize_academic_heatmap()

        # === SUIVI DES MINÉRAUX RÉELS ===
        self.minerals_collected = 0
        self.steps_without_mineral = 0
        self.last_mineral_position = None
        self.mineral_positions = []
        self.concentration_history = []

        # === SUIVI EXPLORATION ===
        self.visited_positions = set()
        self.cleaned_positions = set()
        self.unique_positions_count = 0

        # === COMPTEURS GAUSSIENS ===
        self.gaussian_step = 0
        self.total_steps = 0

        # === STATISTIQUES ===
        self.total_reward = 0.0
        self.academic_reward_total = 0.0
        self.real_reward_total = 0.0
        self.episode_rewards = []

    def _initialize_academic_heatmap(self):
        """Initialise la heatmap académique SPARSE (quelques dépôts concentrés)"""
        height, width = self.grid_size
        # Commencer avec des valeurs très basses (fond = 0)
        heatmap = np.zeros((height, width), dtype=np.float32)

        # Seulement 6 dépôts localisés au lieu de bruit uniforme
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

        # Légère diffusion pour rendre les bords doux
        heatmap = gaussian_filter(heatmap, sigma=1.5)
        return np.clip(heatmap, 0, 1)

    def calculate_reward(self, mineral_concentrations, position,
                         is_new_position=False, has_collision=False,
                         step_count=0, sensor_data=None):
        """
        Calcule la récompense ACADÉMIQUE hybride
        Combine: heatmap simulée + données minérales réelles
        """
        x, y = position
        position_key = (int(x), int(y))
        self.total_steps += 1

        # === 1. RÉCOMPENSE ACADÉMIQUE (heatmap) ===
        academic_reward = self._calculate_academic_reward(position_key, has_collision, mineral_concentrations)

        # === 2. RÉCOMPENSE RÉELLE (minéraux) ===
        real_reward = self._calculate_real_mineral_reward(mineral_concentrations, position_key)

        # === 3. COMBINAISON HYBRIDE (70% réel, 30% académique) ===
        hybrid_reward = 0.7 * real_reward + 0.3 * academic_reward

        # === 4. BONUS STRATÉGIQUES ===
        strategic_bonus = self._calculate_strategic_bonus(position_key, step_count)
        hybrid_reward += strategic_bonus

        # === 5. DIFFUSION GAUSSIENNE ===
        self._update_gaussian_diffusion()

        # === 6. MISE À JOUR ===
        self._update_tracking(position_key, is_new_position, real_reward > 0)
        self.total_reward += hybrid_reward
        self.academic_reward_total += academic_reward
        self.real_reward_total += real_reward

        if np.isnan(hybrid_reward) or np.isinf(hybrid_reward):
            return 0.0

        return hybrid_reward

    def _calculate_academic_reward(self, position_key, has_collision, mineral_concentrations=None):
        """
        Calcule la récompense académique: cpi = Σ(s(j,l) × xi(j,l))
        Si la cellule contient des minéraux REE, la pénalité de revisitation
        ne s'applique pas — le robot doit pouvoir exploiter les gisements.
        """
        x, y = position_key

        if not (0 <= y < self.grid_size[0] and 0 <= x < self.grid_size[1]):
            return self.reward_config['academic_penalty']

        if has_collision:
            return self.reward_config['collision_penalty']

        # Vérifier si la cellule contient des minéraux REE
        max_concentration = 0.0
        if mineral_concentrations:
            max_concentration = max(mineral_concentrations)

        if position_key in self.cleaned_positions:
            # Cellule REE riche : pas de pénalité, reward partiel d'exploitation
            if max_concentration >= self.reward_config['mineral_threshold']:
                return max_concentration * self.reward_config['mineral_base_reward'] * 0.4
            # Cellule vide revisitée : pénalité normale
            return self.reward_config['revisiting_penalty']

        heatmap_value = self.academic_heatmap[y, x]

        if heatmap_value < self.reward_config['clean_threshold']:
            return self.reward_config['academic_penalty']

        academic_reward = heatmap_value * self.reward_config['mineral_base_reward']
        self.academic_heatmap[y, x] *= 0.1
        self.cleaned_positions.add(position_key)

        return academic_reward

    def _calculate_real_mineral_reward(self, concentrations, position_key):
        """Calcule la récompense pour minéraux réels"""
        if not concentrations:
            return 0.0

        max_concentration = max(concentrations) if concentrations else 0.0

        if max_concentration < self.reward_config['mineral_threshold']:
            self.steps_without_mineral += 1
            return 0.0

        mineral_reward = 0.0
        self.steps_without_mineral = 0

        mineral_reward += max_concentration * self.reward_config['mineral_base_reward']
        mineral_reward += max_concentration * self.reward_config['concentration_multiplier']

        if max_concentration > 0.7:
            mineral_reward += self.reward_config['high_concentration_bonus']

        self.minerals_collected += 1
        self.mineral_positions.append(position_key)
        self.concentration_history.append(max_concentration)
        self.last_mineral_position = position_key

        return mineral_reward

    def _calculate_strategic_bonus(self, position_key, step_count):
        """Bonus stratégiques pour comportements intelligents"""
        bonus = 0.0

        if position_key not in self.visited_positions:
            bonus += self.reward_config['exploration_bonus']
            if step_count < 50:
                bonus += self.reward_config['new_zone_bonus']

        coverage = len(self.visited_positions) / (self.grid_size[0] * self.grid_size[1])
        bonus += coverage * self.reward_config['coverage_bonus'] * 1000

        if step_count > 10 and self.minerals_collected > 0:
            efficiency = self.minerals_collected / max(step_count, 1)
            bonus += efficiency * self.reward_config['efficiency_bonus'] * 100

        bonus += self.reward_config['step_penalty']

        return bonus

    def _update_gaussian_diffusion(self):
        """Applique la diffusion gaussienne académique"""
        self.gaussian_step += 1

        if self.gaussian_step % self.reward_config['gaussian_update_freq'] == 0:
            sigma = self.reward_config['gaussian_sigma']
            self.academic_heatmap = gaussian_filter(self.academic_heatmap, sigma=sigma)

            regeneration_rate = 0.05
            mask = np.random.rand(*self.academic_heatmap.shape) < regeneration_rate
            self.academic_heatmap[mask] = np.minimum(1.0, self.academic_heatmap[mask] + 0.1)

            positions_to_remove = []
            for pos in self.cleaned_positions:
                x, y = pos
                if 0 <= y < self.grid_size[0] and 0 <= x < self.grid_size[1]:
                    if np.random.random() < 0.1:
                        positions_to_remove.append(pos)
                        self.academic_heatmap[y, x] = np.random.random() * 0.5

            for pos in positions_to_remove:
                self.cleaned_positions.remove(pos)

    def _update_tracking(self, position_key, is_new_position, found_mineral):
        """Met à jour le suivi"""
        if is_new_position and position_key not in self.visited_positions:
            self.visited_positions.add(position_key)
            self.unique_positions_count += 1

    def get_statistics(self):
        """Retourne les statistiques complètes académiques"""
        avg_concentration = np.mean(self.concentration_history) if self.minerals_collected > 0 else 0.0
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
        """Réinitialise pour un nouvel épisode (conserve l'apprentissage)"""
        self.episode_rewards.append(self.total_reward)
        self.total_reward = 0.0
        self.academic_reward_total = 0.0
        self.real_reward_total = 0.0
        self.visited_positions.clear()
        self.cleaned_positions.clear()
        self.unique_positions_count = 0
        self.steps_without_mineral = 0
        self.total_steps = 0
        self.gaussian_step = 0

        regeneration = np.random.rand(*self.academic_heatmap.shape) * 0.3
        self.academic_heatmap = np.minimum(1.0, self.academic_heatmap + regeneration)

    def get_reward_breakdown(self, position, concentrations):
        """Retourne la décomposition détaillée des récompenses"""
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
            'gaussian_updates': self.gaussian_step // self.reward_config['gaussian_update_freq'],
        }
