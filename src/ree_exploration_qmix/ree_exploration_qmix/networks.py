#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CNNEncoder(nn.Module):
    """Encodeur CNN pour traiter les cartes locales (6 canaux : 4 minéraux + obstacles + exploration)"""

    def __init__(self, input_channels=6, hidden_dim=64, input_size=20):
        super(CNNEncoder, self).__init__()

        # Kernels adaptés à l'entrée input_size×input_size (défaut 20×20) :
        # conv1(k=4,s=2): (20-4)/2+1 = 9  → 9×9
        # conv2(k=3,s=1): (9-3)+1   = 7   → 7×7
        # conv3(k=3,s=1): (7-3)+1   = 5   → 5×5  (64×5×5 = 1600)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=4, stride=2)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3   = nn.BatchNorm2d(64)

        # Calculer dynamiquement la taille de sortie
        self._get_conv_output_size(input_channels, input_size)

        self.fc      = nn.Linear(self.conv_output_size, hidden_dim)
        self.dropout = nn.Dropout(0.2)

    def _get_conv_output_size(self, input_channels, input_size):
        """Calcule la taille de sortie des couches convolutionnelles"""
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_size, input_size)
            x = self.conv1(dummy)
            x = self.conv2(x)
            x = self.conv3(x)
            self.conv_output_size = x.view(1, -1).size(1)

    def forward(self, x):
        """x: (batch, C, H, W)"""
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc(x)))
        return x

class QMixLocalNetwork(nn.Module):
    """Réseau de valeur Q locale pour un agent"""

    def __init__(self, input_shape, num_actions, hidden_dim=64, local_obs_size=20):
        super(QMixLocalNetwork, self).__init__()

        self.num_actions = num_actions

        # Encodeur CNN — utilise la taille réelle de l'observation locale
        self.encoder = CNNEncoder(
            input_channels=input_shape[0],
            hidden_dim=hidden_dim,
            input_size=local_obs_size
        )
        
        # Couche pour la position (normalisée)
        self.position_fc = nn.Linear(2, hidden_dim // 4)
        
        # Réseau de valeur Q
        self.fc1 = nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q_out = nn.Linear(hidden_dim, num_actions)
        
    def forward(self, mineral_map, position):
        """
        Args:
            mineral_map: (batch, C, H, W)
            position: (batch, 2) positions normalisées
        """
        # Encoder la carte minérale
        map_features = self.encoder(mineral_map)
        
        # Encoder la position
        pos_features = F.relu(self.position_fc(position))
        
        # Combiner
        combined = torch.cat([map_features, pos_features], dim=1)
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        q_values = self.q_out(x)
        
        return q_values

class QMixHyperNetwork(nn.Module):
    """Hypernetwork qui génère les poids positifs pour le réseau de mixage"""
    
    def __init__(self, state_dim, hyper_dim, num_agents, hidden_dim):
        super(QMixHyperNetwork, self).__init__()
        
        self.num_agents = num_agents
        self.hyper_dim = hyper_dim
        self.hidden_dim = hidden_dim
        
        # Couches pour générer les poids de la première couche
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hyper_dim),
            nn.ReLU(),
            nn.Linear(hyper_dim, num_agents * hidden_dim)
        )
        
        # Couches pour générer les biais de la première couche
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)
        
        # Couches pour générer les poids de la deuxième couche
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hyper_dim),
            nn.ReLU(),
            nn.Linear(hyper_dim, hidden_dim * 1)  # Sortie vers 1 valeur
        )
        
        # Couches pour générer les biais de la deuxième couche
        self.hyper_b2 = nn.Linear(state_dim, 1)
        
    def forward(self, state):
        """
        Args:
            state: (batch, state_dim) état global
        Returns:
            w1, b1, w2, b2: poids et biais pour le réseau de mixage
        """
        batch_size = state.size(0)
        
        # Générer w1 et forcer la positivité avec un softplus
        w1 = torch.abs(self.hyper_w1(state))
        w1 = w1.view(batch_size, self.num_agents, self.hidden_dim)
        
        # Générer b1
        b1 = self.hyper_b1(state)
        b1 = b1.view(batch_size, 1, self.hidden_dim)
        
        # Générer w2 et forcer la positivité
        w2 = torch.abs(self.hyper_w2(state))
        w2 = w2.view(batch_size, self.hidden_dim, 1)
        
        # Générer b2
        b2 = self.hyper_b2(state)
        b2 = b2.view(batch_size, 1, 1)
        
        return w1, b1, w2, b2

class QMixMixingNetwork(nn.Module):
    """Réseau de mixage monotone QMIX"""
    
    def __init__(self, state_dim, num_agents, hyper_dim=64, hidden_dim=64):
        super(QMixMixingNetwork, self).__init__()
        
        self.num_agents = num_agents
        self.hidden_dim = hidden_dim
        
        # Hypernetwork pour générer les poids
        self.hypernet = QMixHyperNetwork(state_dim, hyper_dim, num_agents, hidden_dim)
        
    def forward(self, q_values, state):
        """
        Args:
            q_values: (batch, num_agents) valeurs Q de chaque agent
            state: (batch, state_dim) état global
        Returns:
            q_tot: (batch, 1) valeur Q totale
        """
        batch_size = q_values.size(0)
        
        # Obtenir les poids de l'hypernetwork
        w1, b1, w2, b2 = self.hypernet(state)
        
        # Première couche
        q_values = q_values.unsqueeze(2)  # (batch, num_agents, 1)
        hidden = torch.bmm(q_values.transpose(1, 2), w1) + b1  # (batch, 1, hidden_dim)
        hidden = F.elu(hidden)
        
        # Deuxième couche
        q_tot = torch.bmm(hidden, w2) + b2  # (batch, 1, 1)
        q_tot = q_tot.squeeze(1).squeeze(1)  # (batch,)
        
        return q_tot

class QMixNetwork(nn.Module):
    """Réseau QMIX complet"""

    def __init__(self, state_shape, num_agents, num_actions,
                 hidden_dim=64, hyper_dim=64, local_obs_size=20):
        super(QMixNetwork, self).__init__()

        self.num_agents = num_agents
        self.num_actions = num_actions

        # Réseau local PARTAGÉ entre tous les agents (comme CleanMARL)
        # Un seul réseau = moins de paramètres + politique homogène entre robots
        self.shared_agent_network = QMixLocalNetwork(
            state_shape, num_actions, hidden_dim, local_obs_size
        )
        
        # Taille de l'état global (carte entière aplatie)
        state_dim = state_shape[0] * state_shape[1] * state_shape[2]
        
        # Réseau de mixage
        self.mixing_network = QMixMixingNetwork(
            state_dim=state_dim,
            num_agents=num_agents,
            hyper_dim=hyper_dim,
            hidden_dim=hidden_dim
        )
        
    def forward(self, mineral_maps, positions, global_state, available_actions=None):
        """
        Args:
            mineral_maps: list of (batch, C, H, W) pour chaque agent
            positions: list of (batch, 2) pour chaque agent
            global_state: (batch, C, H, W) état global complet
            available_actions: optional mask des actions disponibles
        Returns:
            q_tot: (batch,) valeur Q totale
            q_individuals: (batch, num_agents, num_actions) valeurs Q individuelles
        """
        batch_size = mineral_maps[0].size(0)
        
        # Calculer les Q individuels — réseau PARTAGÉ appliqué à chaque agent
        q_individuals = []
        for i in range(self.num_agents):
            q = self.shared_agent_network(mineral_maps[i], positions[i])
            if available_actions is not None:
                q = q.masked_fill(~available_actions[i], -1e10)
            q_individuals.append(q)
        
        # Q_tot via le mixing network avec les Q max (utilisé pour l'inférence/target)
        q_max = torch.stack([q.max(1)[0] for q in q_individuals], dim=1)  # (batch, num_agents)
        state_flat = global_state.view(batch_size, -1)
        q_tot = self.mixing_network(q_max, state_flat)

        # Retourne aussi les Q individuelles (toutes actions) pour la loss
        return q_tot, q_individuals
    
    def get_local_q_values(self, mineral_map, position):
        """Retourne les Q-values via le réseau partagé"""
        with torch.no_grad():
            return self.shared_agent_network(mineral_map, position)

class QMixTargetNetwork(nn.Module):
    """Réseau cible pour QMIX (copie du réseau principal)"""
    
    def __init__(self, qmix_network):
        super(QMixTargetNetwork, self).__init__()
        self.qmix_network = qmix_network
        
    def forward(self, mineral_maps, positions, global_state, available_actions=None):
        return self.qmix_network(mineral_maps, positions, global_state, available_actions)
    
    def update(self, source_network):
        """Met à jour les poids du réseau cible"""
        self.qmix_network.load_state_dict(source_network.state_dict())