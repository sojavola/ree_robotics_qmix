#!/usr/bin/env python3
"""
Geo-ICM : Module de Curiosité Intrinsèque Coopérative pour l'exploration géologique.

article relatifs

  - Pathak et al. (2017) "Curiosity-driven Exploration by Self-Supervised Prediction"
  - Burda et al. (2019) "Large-Scale Study of Curiosity-Driven Learning"
  - Iqbal & Sha (2019) "Actor-Attention-Critic for Multi-Agent RL"

Principe :
  Un forward model partagé prédit les features CNN au pas t+1 à partir des
  features au pas t et de l'action choisie. 
  L'erreur de prédiction constitue
  la récompense de curiosité :  r_curiosity = ||φ̂(s_{t+1}) - φ(s_{t+1})||²

  Le réseau est partagé entre tous les agents (comme le réseau Q local),
  ce qui crée une curiosité *coopérative* : quand un agent explore une zone
  et réduit l'erreur, tous les agents en bénéficient via les poids partagés.

  r_total = r_extrinsic + β * r_curiosity
  β = curiosity_weight (défaut 0.1)

Respect QMIX/CTDE :
  - Le forward model opère sur les features LOCALES (pas l'état global)
  - Il est entraîné de manière centralisée dans le trainer
  - Les agents l'utilisent de manière décentralisée pour le bonus de curiosité
  - La structure QMIX (mixing network, monotonie) est inchangée
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeoICMForwardModel(nn.Module):
    """
    Forward model : prédit les features CNN au pas suivant.

    φ(s_t) + a_t  →  φ̂(s_{t+1})

    Architecture :
        concat(features_64, action_onehot_8) → FC(72, 128) → ReLU
                                              → FC(128, 128) → ReLU
                                              → FC(128, 64)  → φ̂
    ~13k paramètres — léger, n'alourdit pas l'entraînement.
    """

    def __init__(self, feature_dim=64, action_dim=8, hidden_dim=128):
        super(GeoICMForwardModel, self).__init__()

        self.feature_dim = feature_dim
        self.action_dim = action_dim

        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )

    def predict_next_features(self, features_t, action):
        """
        Prédit les features au pas t+1.

        Args:
            features_t: (batch, feature_dim)  — features CNN actuelles
            action:     (batch,)              — actions entières [0..action_dim)
        Returns:
            predicted:  (batch, feature_dim)  — prédiction de φ(s_{t+1})
        """
        action_onehot = F.one_hot(action.long(), self.action_dim).float()
        x = torch.cat([features_t, action_onehot], dim=-1)
        return self.forward_model(x)

    def compute_curiosity_reward(self, features_t, action, features_t1):
        """
        Calcule la récompense de curiosité (erreur de prédiction).

        r_curiosity = ||φ̂(s_{t+1}) - φ(s_{t+1})||²  (moyenne sur feature_dim)

        Args:
            features_t:  (batch, feature_dim) — features CNN au pas t
            action:      (batch,)             — actions choisies
            features_t1: (batch, feature_dim) — features CNN réelles au pas t+1
        Returns:
            curiosity:   (batch,)             — récompense de curiosité par sample
        """
        predicted = self.predict_next_features(features_t, action)
        # MSE par sample (pas par batch) — detach target pour éviter les gradients
        curiosity = F.mse_loss(
            predicted, features_t1.detach(), reduction='none'
        ).mean(dim=-1)
        return curiosity

    def compute_loss(self, features_t, action, features_t1):
        """
        Loss pour entraîner le forward model.

        L_ICM = ||φ̂(s_{t+1}) - φ(s_{t+1})||²

        Args:
            features_t:  (batch, feature_dim)
            action:      (batch,)
            features_t1: (batch, feature_dim)
        Returns:
            loss: scalar
        """
        predicted = self.predict_next_features(features_t, action)
        return F.mse_loss(predicted, features_t1.detach())


class GeoICM(nn.Module):
    """
    Module Geo-ICM complet avec normalisation de la curiosité.

    Intègre :
    - Le forward model
    - Un running mean/std pour normaliser les récompenses de curiosité
      (évite que la curiosité domine ou soit négligeable par rapport à r_ext)
    """

    def __init__(self, feature_dim=64, action_dim=8, hidden_dim=128,
                 curiosity_weight=0.1):
        super(GeoICM, self).__init__()

        self.curiosity_weight = curiosity_weight
        self.forward_model = GeoICMForwardModel(feature_dim, action_dim, hidden_dim)

        # Running statistics pour normalisation
        self.register_buffer('curiosity_mean', torch.tensor(0.0))
        self.register_buffer('curiosity_var', torch.tensor(1.0))
        self.register_buffer('curiosity_count', torch.tensor(0.0))

    def compute_curiosity_reward(self, features_t, action, features_t1,
                                  normalize=True):
        """
        Calcule β * r_curiosity (normalisé optionnellement).

        Args:
            features_t:  (batch, feature_dim) ou (feature_dim,)
            action:      (batch,) ou scalar
            features_t1: (batch, feature_dim) ou (feature_dim,)
            normalize:   bool — normaliser par running stats
        Returns:
            reward:      (batch,) ou scalar — β * curiosité normalisée
        """
        single = features_t.dim() == 1
        if single:
            features_t = features_t.unsqueeze(0)
            action = action.unsqueeze(0) if action.dim() == 0 else action
            features_t1 = features_t1.unsqueeze(0)

        with torch.no_grad():
            raw_curiosity = self.forward_model.compute_curiosity_reward(
                features_t, action, features_t1
            )

            if normalize and self.curiosity_count > 10:
                std = torch.sqrt(self.curiosity_var + 1e-8)
                normalized = (raw_curiosity - self.curiosity_mean) / std
                # Clamp pour éviter les valeurs extrêmes
                normalized = torch.clamp(normalized, -3.0, 3.0)
                reward = self.curiosity_weight * F.relu(normalized)
            else:
                reward = self.curiosity_weight * raw_curiosity

            # Mise à jour des running stats
            self._update_stats(raw_curiosity)

        if single:
            return reward.squeeze(0)
        return reward

    def _update_stats(self, values):
        """Met à jour le running mean/var (Welford online algorithm)."""
        batch_mean = values.mean()
        batch_var = values.var() if values.numel() > 1 else torch.tensor(0.0)
        batch_count = values.numel()

        delta = batch_mean - self.curiosity_mean
        total_count = self.curiosity_count + batch_count

        if total_count > 0:
            self.curiosity_mean = (
                self.curiosity_mean + delta * batch_count / total_count
            )
            self.curiosity_var = (
                (self.curiosity_var * self.curiosity_count +
                 batch_var * batch_count +
                 delta ** 2 * self.curiosity_count * batch_count / total_count)
                / total_count
            )
            self.curiosity_count = total_count

    def compute_loss(self, features_t, action, features_t1):
        """Loss pour entraîner le forward model."""
        return self.forward_model.compute_loss(features_t, action, features_t1)
