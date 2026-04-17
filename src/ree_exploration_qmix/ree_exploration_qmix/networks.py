#!/usr/bin/env python3
"""
Réseaux de neurones QMIX avec 3 contributions PhD :
  1. Multi-Scale CNN Encoder (observation locale + régionale avec attention cross-scale)
  2. GeoCommQMIX (communication apprise entre agents avec gate + attention)
  3. Support Geo-ICM (extraction de features pour le forward model)

Respect QMIX/MARL/CTDE :
  - Réseau Q local PARTAGÉ entre tous les agents
  - Monotonie via torch.abs() sur les poids du mixing network (IGM garanti)
  - État global utilisé UNIQUEMENT dans le mixing network (CTDE)
  - Communication = input supplémentaire aux agents, ne change pas la structure QMIX
  - Multi-scale = encodeur enrichi, ne change pas le mixing network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ═══════════════════════════════════════════════════════════════════════
#  CNN ENCODER (inchangé — brique de base)
# ═══════════════════════════════════════════════════════════════════════

class CNNEncoder(nn.Module):
    """Encodeur CNN pour cartes locales (6 canaux : 4 minéraux + obstacles + exploration)"""

    def __init__(self, input_channels=6, hidden_dim=64, input_size=20):
        super(CNNEncoder, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=4, stride=2)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3   = nn.BatchNorm2d(64)

        self._get_conv_output_size(input_channels, input_size)

        self.fc      = nn.Linear(self.conv_output_size, hidden_dim)
        self.dropout = nn.Dropout(0.2)

    def _get_conv_output_size(self, input_channels, input_size):
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_size, input_size)
            x = self.conv1(dummy)
            x = self.conv2(x)
            x = self.conv3(x)
            self.conv_output_size = x.view(1, -1).size(1)

    def get_feature_maps(self, x):
        """
        Retourne les feature maps spatiales AVANT flatten+FC.

        Utilisé par MultiScaleCNNEncoder pour l'attention cross-scale spatiale.
        Pour un input 20×20 : sortie (batch, 64, 5, 5) → 25 tokens spatiaux.
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x  # (B, 64, H', W')

    def forward(self, x):
        """x: (batch, C, H, W) → (batch, hidden_dim)"""
        x = self.get_feature_maps(x)
        x = self.dropout(F.relu(self.fc(x.reshape(x.size(0), -1))))
        return x


# ═══════════════════════════════════════════════════════════════════════
#  CONTRIBUTION 2 : MULTI-SCALE CNN ENCODER
# ═══════════════════════════════════════════════════════════════════════

class MultiScaleCNNEncoder(nn.Module):
    """
    Encodeur CNN multi-échelle avec attention cross-scale.

    Deux échelles :
      - Locale  (20×20 à résolution 1:1) — détails fins des minéraux
      - Régionale (60×60 sous-échantillonnée à 20×20) — contexte large

    Fusion par cross-scale attention :
      Q = features_locales, K = V = features_régionales
      → Le réseau apprend à pondérer l'information régionale
         en fonction du contexte local.

    Référence : Chen et al. (2020) "Multi-Scale Feature Aggregation"
    """

    def __init__(self, input_channels=6, hidden_dim=64, local_size=20,
                 regional_size=20, num_heads=4):
        super(MultiScaleCNNEncoder, self).__init__()

        self.hidden_dim = hidden_dim

        # Encodeur local (fine-grained, 20×20)
        self.local_encoder = CNNEncoder(input_channels, hidden_dim, local_size)

        # Encodeur régional (coarse-grained, 60×60 downsampled → 20×20)
        self.regional_encoder = CNNEncoder(input_channels, hidden_dim, regional_size)

        # Cross-scale attention : local attend to regional
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # Fusion FC : concat(local, attended_regional) → hidden_dim
        self.fusion_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, local_map, regional_map):
        """
        Args:
            local_map:    (batch, C, 20, 20) — observation locale
            regional_map: (batch, C, 20, 20) — observation régionale (déjà downsampled)
        Returns:
            fused_features: (batch, hidden_dim)

        Attention cross-scale SPATIALE :
          - conv3 produit (B, 64, 5, 5) = 25 tokens spatiaux pour chaque échelle
          - Q = 25 tokens locaux, K = V = 25 tokens régionaux
          - Chaque position locale apprend à extraire le contexte régional pertinent
          - Global average pooling pour agréger les 25 tokens → vecteur (B, 64)
        """
        # Extraire les feature maps spatiales (B, 64, 5, 5)
        local_maps = self.local_encoder.get_feature_maps(local_map)
        regional_maps = self.regional_encoder.get_feature_maps(regional_map)

        B, C, H, W = local_maps.shape  # C=64, H=W=5 → 25 tokens spatiaux

        # Reshape : (B, 64, 5, 5) → (B, 25, 64) — séquences de tokens spatiaux
        local_seq = local_maps.view(B, C, -1).transpose(1, 2)     # (B, 25, 64)
        regional_seq = regional_maps.view(B, C, -1).transpose(1, 2)  # (B, 25, 64)

        # Cross-scale attention :
        # Q = tokens locaux (25 positions), K = V = tokens régionaux (25 positions)
        # Chaque token local attend à toutes les positions régionales
        attended, _ = self.cross_attention(local_seq, regional_seq, regional_seq)
        attended = self.attn_norm(attended + local_seq)  # residual (B, 25, 64)

        # Agrégation : global average pooling → (B, 64)
        local_global = local_seq.mean(dim=1)    # (B, 64)
        attended_global = attended.mean(dim=1)  # (B, 64)

        # Fusion
        fused = torch.cat([local_global, attended_global], dim=-1)  # (B, 128)
        return self.fusion_fc(fused)  # (B, hidden_dim)

    def encode_local_only(self, local_map):
        """Encode uniquement l'observation locale (fallback si pas de régionale)."""
        return self.local_encoder(local_map)


# ═══════════════════════════════════════════════════════════════════════
#  CONTRIBUTION 3 : GeoCommQMIX — MODULE DE COMMUNICATION
# ═══════════════════════════════════════════════════════════════════════

class CommModule(nn.Module):
    """
    Module de communication apprise entre agents (GeoCommQMIX).

    Architecture :
      1. Message Encoder : features (64) → message (32)
      2. Attention : l'agent attend aux messages des N-1 autres agents
      3. Gate : σ(W·[features, attended_msg]) ∈ [0,1] — contrôle l'influence
      4. Output : features + gate * transform(attended_msg)

    Respect CTDE :
      - Pendant l'entraînement : tous les messages calculés dans un forward pass
      - Pendant l'exécution : messages échangés via topics ROS2
      - Le gate permet à l'agent d'ignorer les messages non pertinents

    Référence : Sukhbaatar et al. (2016) "Learning Multiagent Communication"
                Das et al. (2019) "TarMAC: Targeted Multi-Agent Communication"
    """

    def __init__(self, feature_dim=64, comm_dim=32, num_agents=4):
        super(CommModule, self).__init__()

        self.feature_dim = feature_dim
        self.comm_dim = comm_dim
        self.num_agents = num_agents

        # 1. Message Encoder : features → compact message
        self.message_encoder = nn.Sequential(
            nn.Linear(feature_dim, comm_dim),
            nn.ReLU()
        )

        # 2. Attention : pondérer les messages reçus
        self.msg_attention = nn.MultiheadAttention(
            comm_dim, num_heads=4, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(comm_dim)

        # 3. Gate : contrôle combien la communication influence l'agent
        self.gate = nn.Sequential(
            nn.Linear(feature_dim + comm_dim, 1),
            nn.Sigmoid()
        )

        # 4. Output projection : message → feature space
        self.output_fc = nn.Linear(comm_dim, feature_dim)

    def encode_message(self, features):
        """
        Encode les features en un message compact.

        Args:
            features: (batch, feature_dim)
        Returns:
            message:  (batch, comm_dim)
        """
        return self.message_encoder(features)

    def process_messages(self, own_features, received_messages):
        """
        Traite les messages reçus et enrichit les features de l'agent.

        Args:
            own_features:      (batch, feature_dim)
            received_messages: (batch, N-1, comm_dim)  — messages des autres agents
        Returns:
            enhanced_features: (batch, feature_dim)
        """
        # Query = propre message encodé
        own_msg = self.message_encoder(own_features).unsqueeze(1)  # (B, 1, comm_dim)

        # Attention sur les messages reçus
        attended, _ = self.msg_attention(
            own_msg, received_messages, received_messages
        )
        attended = self.attn_norm(attended.squeeze(1))  # (B, comm_dim)

        # Gate : contrôler l'influence de la communication
        gate_input = torch.cat([own_features, attended], dim=-1)
        gate_value = self.gate(gate_input)  # (B, 1)

        # Combiner : features + gate * projected_message
        comm_features = self.output_fc(attended)  # (B, feature_dim)
        return own_features + gate_value * comm_features


# ═══════════════════════════════════════════════════════════════════════
#  RÉSEAU Q LOCAL (MODIFIÉ — supporte multi-scale + communication)
# ═══════════════════════════════════════════════════════════════════════

class QMixLocalNetwork(nn.Module):
    """
    Réseau de valeur Q locale pour un agent.

    Supporte les 3 contributions via des flags :
      - use_multi_scale : utilise MultiScaleCNNEncoder au lieu de CNNEncoder
      - use_comm : intègre le CommModule pour enrichir les features

    Méthodes clés :
      - encode()  : extraction de features (utilisé par ICM + communication)
      - forward() : Q-values complètes (features + position + comm → Q)
    """

    def __init__(self, input_shape, num_actions, hidden_dim=64,
                 local_obs_size=20, use_multi_scale=False,
                 use_comm=False, comm_dim=32, num_agents=4):
        super(QMixLocalNetwork, self).__init__()

        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.use_multi_scale = use_multi_scale
        self.use_comm = use_comm

        # === Encodeur CNN (contribution 2 : multi-scale) ===
        if use_multi_scale:
            self.encoder = MultiScaleCNNEncoder(
                input_channels=input_shape[0],
                hidden_dim=hidden_dim,
                local_size=local_obs_size,
                regional_size=local_obs_size  # Régional downsampled à la même taille
            )
        else:
            self.encoder = CNNEncoder(
                input_channels=input_shape[0],
                hidden_dim=hidden_dim,
                input_size=local_obs_size
            )

        # === Communication (contribution 3 : GeoCommQMIX) ===
        if use_comm:
            self.comm_module = CommModule(hidden_dim, comm_dim, num_agents)

        # Position encoding (normalisée)
        self.position_fc = nn.Linear(2, hidden_dim // 4)

        # Réseau de valeur Q
        combined_dim = hidden_dim + hidden_dim // 4
        self.fc1 = nn.Linear(combined_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q_out = nn.Linear(hidden_dim, num_actions)

    def encode(self, mineral_map, regional_map=None):
        """
        Extrait les features CNN (utilisé par ICM + communication).

        Args:
            mineral_map:  (batch, C, H, W) — observation locale
            regional_map: (batch, C, H, W) — observation régionale (optionnel)
        Returns:
            features: (batch, hidden_dim)
        """
        if self.use_multi_scale and regional_map is not None:
            return self.encoder(mineral_map, regional_map)
        elif self.use_multi_scale:
            return self.encoder.encode_local_only(mineral_map)
        else:
            return self.encoder(mineral_map)

    def forward(self, mineral_map, position, regional_map=None,
                received_messages=None):
        """
        Forward complet : encode + (comm) + Q-values.

        Args:
            mineral_map:       (batch, C, H, W)
            position:          (batch, 2)
            regional_map:      (batch, C, H, W) optionnel — multi-scale
            received_messages: (batch, N-1, comm_dim) optionnel — communication
        Returns:
            q_values: (batch, num_actions)
        """
        # Encoder l'observation
        features = self.encode(mineral_map, regional_map)

        # Communication : enrichir les features avec les messages reçus
        if self.use_comm and received_messages is not None:
            features = self.comm_module.process_messages(features, received_messages)

        # Position
        pos_features = F.relu(self.position_fc(position))

        # Q-values
        combined = torch.cat([features, pos_features], dim=1)
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        return self.q_out(x)


# ═══════════════════════════════════════════════════════════════════════
#  ENCODEUR D'ÉTAT GLOBAL — remplace flatten(60k) par CNN(→256)
# ═══════════════════════════════════════════════════════════════════════

class StateEncoder(nn.Module):
    """
    Encode l'état global (C, 100, 100) → vecteur compact (256 dims).

    Remplace le flatten naïf 6×100×100 = 60 000 dimensions :
      - Avant : Linear(60 000, 64) = ~3.8M params pour la 1ère couche hypernetwork
      - Après : CNN → 256 dims → ~164k params total
    Gain : -23M paramètres dans l'hypernetwork, entraînement 5-10× plus rapide.
    """
    STATE_ENCODED_DIM = 256

    def __init__(self, input_channels=6):
        super(StateEncoder, self).__init__()
        # Conv1(k=8,s=4) : 100→24, Conv2(k=4,s=2) : 24→11, Conv3(k=3,s=1) : 11→9
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 100, 100)
            x = self.conv3(self.conv2(self.conv1(dummy)))
            conv_out = x.view(1, -1).size(1)  # 64×9×9 = 5184
        self.fc = nn.Linear(conv_out, self.STATE_ENCODED_DIM)

    def forward(self, state):
        """state: (batch, C, H, W) → (batch, 256)"""
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return F.relu(self.fc(x.reshape(x.size(0), -1)))


# ═══════════════════════════════════════════════════════════════════════
#  HYPERNETWORK + MIXING NETWORK (inchangés — monotonie QMIX)
# ═══════════════════════════════════════════════════════════════════════

class QMixHyperNetwork(nn.Module):
    """Hypernetwork : génère les poids POSITIFS pour le mixing network (IGM)."""

    def __init__(self, state_dim, hyper_dim, num_agents, hidden_dim):
        super(QMixHyperNetwork, self).__init__()

        self.num_agents = num_agents
        self.hyper_dim = hyper_dim
        self.hidden_dim = hidden_dim

        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hyper_dim),
            nn.ReLU(),
            nn.Linear(hyper_dim, num_agents * hidden_dim)
        )
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)

        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hyper_dim),
            nn.ReLU(),
            nn.Linear(hyper_dim, hidden_dim * 1)
        )
        self.hyper_b2 = nn.Linear(state_dim, 1)

    def forward(self, state):
        batch_size = state.size(0)

        w1 = torch.abs(self.hyper_w1(state))
        w1 = w1.view(batch_size, self.num_agents, self.hidden_dim)
        b1 = self.hyper_b1(state).view(batch_size, 1, self.hidden_dim)

        w2 = torch.abs(self.hyper_w2(state))
        w2 = w2.view(batch_size, self.hidden_dim, 1)
        b2 = self.hyper_b2(state).view(batch_size, 1, 1)

        return w1, b1, w2, b2


class QMixMixingNetwork(nn.Module):
    """Réseau de mixage monotone QMIX (∂Q_tot/∂Q_i ≥ 0)."""

    def __init__(self, state_dim, num_agents, hyper_dim=64, hidden_dim=64):
        super(QMixMixingNetwork, self).__init__()

        self.num_agents = num_agents
        self.hidden_dim = hidden_dim
        self.hypernet = QMixHyperNetwork(state_dim, hyper_dim, num_agents, hidden_dim)

    def forward(self, q_values, state):
        w1, b1, w2, b2 = self.hypernet(state)

        q_values = q_values.unsqueeze(2)
        hidden = torch.bmm(q_values.transpose(1, 2), w1) + b1
        hidden = F.elu(hidden)

        q_tot = torch.bmm(hidden, w2) + b2
        q_tot = q_tot.squeeze(1).squeeze(1)
        return q_tot


# ═══════════════════════════════════════════════════════════════════════
#  RÉSEAU QMIX COMPLET (MODIFIÉ — communication-aware training)
# ═══════════════════════════════════════════════════════════════════════

class QMixNetwork(nn.Module):
    """
    Réseau QMIX complet avec support multi-scale + communication.

    Pendant l'entraînement centralisé (CTDE) :
      1. Encoder les observations de TOUS les agents
      2. Communication : échanger des messages entre agents (N rounds)
      3. Calculer les Q-values individuelles enrichies
      4. Mixer via le mixing network (monotone, IGM)

    Pendant l'exécution décentralisée :
      - Chaque agent utilise get_local_q_values() avec ses propres observations
        et les messages reçus via ROS2
    """

    def __init__(self, state_shape, num_agents, num_actions,
                 hidden_dim=64, hyper_dim=64, local_obs_size=20,
                 use_multi_scale=False, use_comm=False,
                 comm_dim=32, num_comm_rounds=1):
        super(QMixNetwork, self).__init__()

        self.num_agents = num_agents
        self.num_actions = num_actions
        self.use_comm = use_comm
        self.num_comm_rounds = num_comm_rounds

        # Réseau local PARTAGÉ (avec multi-scale + comm)
        self.shared_agent_network = QMixLocalNetwork(
            state_shape, num_actions, hidden_dim, local_obs_size,
            use_multi_scale=use_multi_scale,
            use_comm=use_comm,
            comm_dim=comm_dim,
            num_agents=num_agents
        )

        # Encodeur d'état global CNN : (C,100,100) → 256 dims
        self.state_encoder = StateEncoder(input_channels=state_shape[0])

        # Réseau de mixage (INCHANGÉ — monotonie QMIX)
        self.mixing_network = QMixMixingNetwork(
            state_dim=StateEncoder.STATE_ENCODED_DIM,  # 256 au lieu de 60 000
            num_agents=num_agents,
            hyper_dim=hyper_dim,
            hidden_dim=hidden_dim
        )

    def forward(self, mineral_maps, positions, global_state,
                regional_maps=None, available_actions=None):
        """
        Forward centralisé (entraînement CTDE).

        Args:
            mineral_maps:     list[num_agents] of (batch, C, H, W)
            positions:        list[num_agents] of (batch, 2)
            global_state:     (batch, C, H, W) — état global
            regional_maps:    list[num_agents] of (batch, C, H, W) — optionnel
            available_actions: optionnel
        Returns:
            q_tot:         (batch,)
            q_individuals: list[num_agents] of (batch, num_actions)
        """
        batch_size = mineral_maps[0].size(0)
        agent_net = self.shared_agent_network

        # ── Étape 1 : Encoder les observations de tous les agents ──────
        all_features = []
        for i in range(self.num_agents):
            reg = regional_maps[i] if regional_maps is not None else None
            feat = agent_net.encode(mineral_maps[i], reg)
            all_features.append(feat)

        # ── Étape 2 : Communication (si activée) ──────────────────────
        if self.use_comm and agent_net.use_comm:
            for _ in range(self.num_comm_rounds):
                # Encoder les messages de tous les agents
                messages = [
                    agent_net.comm_module.encode_message(f)
                    for f in all_features
                ]
                messages_tensor = torch.stack(messages, dim=1)  # (B, N, comm_dim)

                # Chaque agent traite les messages des N-1 autres
                new_features = []
                for i in range(self.num_agents):
                    # Exclure son propre message
                    other_msgs = torch.cat([
                        messages_tensor[:, :i],
                        messages_tensor[:, i+1:]
                    ], dim=1)  # (B, N-1, comm_dim)

                    enhanced = agent_net.comm_module.process_messages(
                        all_features[i], other_msgs
                    )
                    new_features.append(enhanced)
                all_features = new_features

        # ── Étape 3 : Q-values individuelles ──────────────────────────
        q_individuals = []
        for i in range(self.num_agents):
            pos_features = F.relu(agent_net.position_fc(positions[i]))
            combined = torch.cat([all_features[i], pos_features], dim=1)
            x = F.relu(agent_net.fc1(combined))
            x = F.relu(agent_net.fc2(x))
            q = agent_net.q_out(x)

            if available_actions is not None:
                q = q.masked_fill(~available_actions[i], -1e10)
            q_individuals.append(q)

        # ── Étape 4 : Mixing (monotone, IGM) ─────────────────────────
        q_max = torch.stack(
            [q.max(1)[0] for q in q_individuals], dim=1
        )
        state_encoded = self.encode_state(global_state)  # (B, 256)
        q_tot = self.mixing_network(q_max, state_encoded)

        return q_tot, q_individuals

    def encode_state(self, global_state):
        """Encode l'état global : (B, C, H, W) → (B, 256)."""
        return self.state_encoder(global_state)

    def extract_all_features(self, mineral_maps, regional_maps=None):
        """
        Extrait les features CNN de tous les agents (pour entraînement ICM).

        Args:
            mineral_maps:  list[num_agents] of (batch, C, H, W)
            regional_maps: list[num_agents] of (batch, C, H, W) optionnel
        Returns:
            features: list[num_agents] of (batch, hidden_dim)
        """
        features = []
        for i in range(self.num_agents):
            reg = regional_maps[i] if regional_maps is not None else None
            feat = self.shared_agent_network.encode(mineral_maps[i], reg)
            features.append(feat)
        return features

    def get_local_q_values(self, mineral_map, position, regional_map=None,
                           received_messages=None):
        """
        Q-values pour exécution décentralisée (un seul agent).

        Args:
            mineral_map:       (1, C, H, W)
            position:          (1, 2)
            regional_map:      (1, C, H, W) optionnel
            received_messages: (1, N-1, comm_dim) optionnel
        Returns:
            q_values: (1, num_actions)
        """
        with torch.no_grad():
            return self.shared_agent_network(
                mineral_map, position, regional_map, received_messages
            )

    def encode_message(self, features):
        """Encode un message pour la communication ROS2 (exécution)."""
        if self.use_comm:
            with torch.no_grad():
                return self.shared_agent_network.comm_module.encode_message(features)
        return None


class QMixTargetNetwork(nn.Module):
    """Réseau cible pour QMIX (copie du réseau principal)."""

    def __init__(self, qmix_network):
        super(QMixTargetNetwork, self).__init__()
        self.qmix_network = qmix_network

    def forward(self, mineral_maps, positions, global_state,
                regional_maps=None, available_actions=None):
        return self.qmix_network(
            mineral_maps, positions, global_state,
            regional_maps, available_actions
        )

    def update(self, source_network):
        self.qmix_network.load_state_dict(source_network.state_dict())
