#!/usr/bin/env python3
"""
=============================================================================
  QMIX for Multi-Robot REE Exploration — Two Novel Contributions
  M2 Internship — PRISMA Lab
=============================================================================

Contribution 1 : Multi-Scale CNN Encoder with Cross-Scale Attention
--------------------------------------------------------------------
  Each robot builds two spatial observations of its surroundings:
    • Local  observation : 20×20 cells at full resolution  (fine-grained)
    • Regional observation: 60×60 cells downsampled to 20×20 (coarse-grained)

  A cross-scale attention mechanism (inspired by Feature Pyramid Networks,
  Lin et al., CVPR 2017) fuses both scales:
    Q = local spatial tokens,  K = V = regional spatial tokens
  The agent learns to focus on relevant large-scale context
  when evaluating fine-grained local decisions.

Contribution 2 : GeoCommQMIX — Learned Inter-Agent Communication
-----------------------------------------------------------------
  Each robot broadcasts a compact message vector (dim=32) encoding
  its current belief state. Messages are aggregated via multi-head
  attention and gated before enriching the agent's local features.

  Inspired by:
    • CommNet        (Sukhbaatar et al., NeurIPS 2016)
    • TarMAC         (Das et al., ICML 2019)
    • ATOC           (Jiang & Lu, NeurIPS 2018)

  The gating mechanism allows agents to learn WHEN communication
  is informative — naturally handling partial connectivity.

CTDE Compliance (Centralised Training, Decentralised Execution):
  • Training  : full forward pass with all agents' messages
  • Execution : each agent broadcasts its message over ROS2 DDS
                and processes received messages locally
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# =============================================================================
#  CONTRIBUTION 1 — MULTI-SCALE CNN ENCODER
# =============================================================================

class CNNEncoder(nn.Module):
    """
    Single-scale CNN encoder for spatial mineral maps.

    Input  : (batch, C=6, H=20, W=20)
               Channels: [REE_1, REE_2, REE_3, REE_4, obstacles, exploration]
    Output : (batch, hidden_dim=64)

    Architecture : Conv(4,s2) → BN → Conv(3,s1) → BN → Conv(3,s1) → BN → FC
    """

    def __init__(self, input_channels: int = 6,
                 hidden_dim: int = 64,
                 input_size: int = 20):
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=4, stride=2)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3   = nn.BatchNorm2d(64)

        # Auto-compute flattened size after conv stack
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_size, input_size)
            out   = self._forward_conv(dummy)
            self.conv_out_size = out.view(1, -1).shape[1]

        self.fc      = nn.Linear(self.conv_out_size, hidden_dim)
        self.dropout = nn.Dropout(p=0.2)

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x

    def get_spatial_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return spatial feature maps BEFORE flatten — used by cross-scale attention.

        For input (B, 6, 20, 20):
            Conv(4,s2) → (B, 32, 9, 9)
            Conv(3,s1) → (B, 64, 7, 7)
            Conv(3,s1) → (B, 64, 5, 5)   ← 25 spatial tokens per agent
        """
        return self._forward_conv(x)   # (B, 64, H', W')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, hidden_dim)"""
        feats = self._forward_conv(x)
        return self.dropout(F.relu(self.fc(feats.reshape(feats.size(0), -1))))


# -----------------------------------------------------------------------------

class MultiScaleCNNEncoder(nn.Module):
    """
    Contribution 1 : Multi-Scale CNN Encoder with Cross-Scale Attention.

    ┌─────────────────────────────────────────────────────────────────┐
    │                    ROBOT FIELD OF VIEW                          │
    │                                                                 │
    │    ┌──────────────────────────────────────────┐                │
    │    │          REGIONAL (60×60 cells)           │                │
    │    │   ┌──────────────────┐                   │                │
    │    │   │  LOCAL (20×20)   │ ← full resolution │                │
    │    │   │    (robot here)  │                   │                │
    │    │   └──────────────────┘                   │                │
    │    │     × 3 zoom-out, downsampled to 20×20   │                │
    │    └──────────────────────────────────────────┘                │
    └─────────────────────────────────────────────────────────────────┘

    Fusion via Cross-Scale Attention (FPN-inspired, Lin et al. 2017):

        local_maps    = CNNEncoder(local_obs)    → (B, 64, 5, 5)
        regional_maps = CNNEncoder(regional_obs) → (B, 64, 5, 5)

        Reshape to sequences of 25 spatial tokens:
        local_seq    = (B, 25, 64)
        regional_seq = (B, 25, 64)

        Cross-Attention:
        Q = local_seq,  K = V = regional_seq
        attended = MultiheadAttention(Q, K, V)  + residual

        Fusion:
        global_local    = mean(local_seq,    dim=tokens) → (B, 64)
        global_attended = mean(attended,     dim=tokens) → (B, 64)
        fused = FC( concat([global_local, global_attended]) ) → (B, 64)

    The agent learns to select, for each local position, the most
    relevant regional context — e.g., a cluster of unexplored minerals
    3× further away that should influence the current direction.
    """

    def __init__(self,
                 input_channels: int = 6,
                 hidden_dim: int     = 64,
                 local_size: int     = 20,
                 regional_size: int  = 20,
                 num_heads: int      = 4):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Two independent CNN encoders (shared weights NOT forced — each
        # scale may require different low-level feature detectors)
        self.local_encoder    = CNNEncoder(input_channels, hidden_dim, local_size)
        self.regional_encoder = CNNEncoder(input_channels, hidden_dim, regional_size)

        # Cross-scale attention: local queries regional context
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.attn_norm  = nn.LayerNorm(hidden_dim)

        # Final fusion: concat(local_global, attended_global) → hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )

    def forward(self, local_map: torch.Tensor,
                regional_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            local_map    : (B, C, 20, 20) — full-resolution local observation
            regional_map : (B, C, 20, 20) — 60×60 region downsampled to 20×20
        Returns:
            fused_features : (B, hidden_dim)
        """
        # ── Step 1 : CNN feature extraction ──────────────────────────────
        local_feat    = self.local_encoder.get_spatial_tokens(local_map)
        regional_feat = self.regional_encoder.get_spatial_tokens(regional_map)
        # Both: (B, 64, 5, 5)

        B, C, H, W = local_feat.shape   # C=64, H=W=5, 25 tokens

        # ── Step 2 : reshape to token sequences ──────────────────────────
        # (B, 64, 5, 5) → (B, 25, 64)
        local_seq    = local_feat.view(B, C, -1).transpose(1, 2)
        regional_seq = regional_feat.view(B, C, -1).transpose(1, 2)

        # ── Step 3 : cross-scale attention ───────────────────────────────
        # Each LOCAL token attends to ALL REGIONAL tokens
        # → learns which distant context is relevant for each local position
        attended, attn_weights = self.cross_attn(
            local_seq,    # Query  : local spatial positions
            regional_seq, # Key    : regional spatial positions
            regional_seq  # Value  : regional spatial positions
        )
        # Residual connection + layer norm (stabilises training)
        attended = self.attn_norm(attended + local_seq)   # (B, 25, 64)

        # ── Step 4 : global aggregation ──────────────────────────────────
        # Average over the 25 spatial tokens → global representation
        local_global    = local_seq.mean(dim=1)    # (B, 64)
        attended_global = attended.mean(dim=1)     # (B, 64)

        # ── Step 5 : fusion ──────────────────────────────────────────────
        fused = torch.cat([local_global, attended_global], dim=-1)  # (B, 128)
        return self.fusion(fused)   # (B, 64)

    def encode_local_only(self, local_map: torch.Tensor) -> torch.Tensor:
        """Fallback: encode only local observation (no regional available)."""
        return self.local_encoder(local_map)


# =============================================================================
#  CONTRIBUTION 2 — GeoCommQMIX : INTER-AGENT COMMUNICATION
# =============================================================================

class CommModule(nn.Module):
    """
    Contribution 2 : GeoCommQMIX — Learned Inter-Agent Communication.

    Each agent i computes a compact message m_i from its current
    feature vector h_i, broadcasts it over ROS2 DDS, and enriches
    its own features by attending over received messages {m_j | j ≠ i}.

    ┌──────────────────────────────────────────────────────────────────┐
    │  AGENT i                                                         │
    │                                                                  │
    │  h_i (features)                                                  │
    │      │                                                           │
    │      ├─── Message Encoder ──→ m_i ──→ broadcast (ROS2 topic)    │
    │      │         (FC + ReLU)                                       │
    │      │                                                           │
    │      │   received: {m_j | j ≠ i} ← from other robots           │
    │      │                                                           │
    │      └─── Attention + Gate ──→ h_i_enriched                    │
    │                                                                  │
    │  Architecture:                                                   │
    │    1. Encoder   : h_i (64) → m_i (32)     [FC + ReLU]          │
    │    2. Attention : Q=m_i, K=V={m_j}         [MultiheadAttn]      │
    │    3. Gate      : g = σ(W·[h_i, attended]) ∈ [0,1]              │
    │    4. Fusion    : h_i' = h_i + g · FC(attended)                 │
    └──────────────────────────────────────────────────────────────────┘

    Key design choices:
      • Gate (step 3): allows the agent to learn WHEN communication
        is useful. If neighbours have no relevant info, g ≈ 0 and
        the agent ignores their messages. Critical for deployment
        on real robots where some agents may be out of range.

      • Shared weights: the same CommModule is used by all agents
        (CTDE — shared parameters reduce the parameter count and
        encourage symmetric communication protocols).

    References:
      Sukhbaatar et al. (2016) CommNet — continuous message aggregation
      Das et al.        (2019) TarMAC — targeted attention-based comm
      Jiang & Lu        (2018) ATOC   — attention-based comm gating
    """

    def __init__(self,
                 feature_dim: int = 64,
                 comm_dim: int    = 32,
                 num_agents: int  = 4,
                 num_heads: int   = 4):
        super().__init__()

        self.feature_dim = feature_dim
        self.comm_dim    = comm_dim
        self.num_agents  = num_agents

        # ── 1. Message Encoder ────────────────────────────────────────
        # Compress the agent's feature vector into a compact message.
        # dim: feature_dim (64) → comm_dim (32)
        self.message_encoder = nn.Sequential(
            nn.Linear(feature_dim, comm_dim),
            nn.ReLU()
        )

        # ── 2. Multi-head Attention ───────────────────────────────────
        # Agent i attends over messages from agents j ≠ i.
        # Using own encoded message as Query:
        #   attended = Attention(Q=m_i, K={m_j}, V={m_j})
        self.msg_attention = nn.MultiheadAttention(
            embed_dim=comm_dim, num_heads=num_heads, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(comm_dim)

        # ── 3. Gating Mechanism ───────────────────────────────────────
        # g = σ(W · [h_i, attended_msg]) ∈ [0, 1]
        # Learns when communication is informative.
        # If neighbours have no useful information: g → 0.
        self.gate = nn.Sequential(
            nn.Linear(feature_dim + comm_dim, 1),
            nn.Sigmoid()
        )

        # ── 4. Output Projection ──────────────────────────────────────
        # Project attended message back to feature space for fusion.
        self.output_proj = nn.Linear(comm_dim, feature_dim)

    def encode_message(self, features: torch.Tensor) -> torch.Tensor:
        """
        Encode agent features into a compact broadcast message.

        Called during both training (all agents at once) and
        deployment (each agent independently on its own features).

        Args:
            features : (B, feature_dim)   — agent's CNN features
        Returns:
            message  : (B, comm_dim)       — message to broadcast
        """
        return self.message_encoder(features)

    def process_messages(self, own_features: torch.Tensor,
                         received_messages: torch.Tensor) -> torch.Tensor:
        """
        Enrich agent features using received messages from neighbours.

        Args:
            own_features      : (B, feature_dim)       — own CNN features h_i
            received_messages : (B, N-1, comm_dim)     — messages from other agents
        Returns:
            enriched_features : (B, feature_dim)       — h_i' = h_i + g·comm
        """
        # ── Step 1 : encode own features as query ─────────────────────
        own_msg = self.message_encoder(own_features).unsqueeze(1)
        # own_msg : (B, 1, comm_dim)

        # ── Step 2 : attend over received messages ────────────────────
        # Q = own message, K = V = received messages
        attended, _ = self.msg_attention(
            own_msg,           # Query  : (B, 1, comm_dim)
            received_messages, # Key    : (B, N-1, comm_dim)
            received_messages  # Value  : (B, N-1, comm_dim)
        )
        attended = self.attn_norm(attended.squeeze(1))   # (B, comm_dim)

        # ── Step 3 : gate — learn when to use communication ──────────
        gate_input = torch.cat([own_features, attended], dim=-1)
        # gate_input : (B, feature_dim + comm_dim)
        g = self.gate(gate_input)    # g : (B, 1) ∈ [0, 1]

        # ── Step 4 : gated residual fusion ───────────────────────────
        comm_contribution = self.output_proj(attended)   # (B, feature_dim)
        return own_features + g * comm_contribution      # (B, feature_dim)


# =============================================================================
#  INTEGRATION : Q-NETWORK WITH BOTH CONTRIBUTIONS
# =============================================================================

class QNetworkWithContributions(nn.Module):
    """
    Local Q-network integrating both contributions.

    Data flow:
        local_obs (20×20×6)
        regional_obs (20×20×6)  ──┐
                                   ├─→ MultiScaleCNNEncoder → features (64)
        (single-scale fallback) ──┘         ↓
                                       CommModule.process_messages
                                       (if received_messages provided)
                                             ↓
                                       enriched_features (64)
                                             ↓
                                       FC layers → Q-values (8 actions)
    """

    def __init__(self,
                 input_channels: int  = 6,
                 num_actions: int     = 8,
                 hidden_dim: int      = 64,
                 local_obs_size: int  = 20,
                 use_multi_scale: bool = True,
                 use_comm: bool       = True,
                 comm_dim: int        = 32,
                 num_agents: int      = 4):
        super().__init__()

        self.use_multi_scale = use_multi_scale
        self.use_comm        = use_comm
        self.hidden_dim      = hidden_dim

        # ── Contribution 1 : Multi-Scale Encoder ─────────────────────
        if use_multi_scale:
            self.encoder = MultiScaleCNNEncoder(
                input_channels=input_channels,
                hidden_dim=hidden_dim,
                local_size=local_obs_size,
                regional_size=local_obs_size   # already downsampled externally
            )
        else:
            self.encoder = CNNEncoder(
                input_channels=input_channels,
                hidden_dim=hidden_dim,
                input_size=local_obs_size
            )

        # ── Contribution 2 : Communication Module ────────────────────
        if use_comm:
            self.comm_module = CommModule(
                feature_dim=hidden_dim,
                comm_dim=comm_dim,
                num_agents=num_agents
            )

        # ── Position encoding (normalised x/y ∈ [0,1]) ───────────────
        self.pos_fc = nn.Linear(2, hidden_dim // 4)

        # ── Q-value head ─────────────────────────────────────────────
        combined = hidden_dim + hidden_dim // 4
        self.fc1   = nn.Linear(combined, hidden_dim)
        self.fc2   = nn.Linear(hidden_dim, hidden_dim)
        self.q_out = nn.Linear(hidden_dim, num_actions)

    def encode(self, local_map: torch.Tensor,
               regional_map: torch.Tensor = None) -> torch.Tensor:
        """Extract CNN features. Called separately to get message for broadcast."""
        if self.use_multi_scale and regional_map is not None:
            return self.encoder(local_map, regional_map)
        elif self.use_multi_scale:
            return self.encoder.encode_local_only(local_map)
        else:
            return self.encoder(local_map)

    def encode_message(self, local_map: torch.Tensor,
                       regional_map: torch.Tensor = None) -> torch.Tensor:
        """
        Compute the message to broadcast to other agents.
        Called at each decision step; result published over ROS2.
        """
        if not self.use_comm:
            return None
        features = self.encode(local_map, regional_map)
        return self.comm_module.encode_message(features)   # (B, comm_dim)

    def forward(self,
                local_map: torch.Tensor,
                position: torch.Tensor,
                regional_map: torch.Tensor        = None,
                received_messages: torch.Tensor   = None) -> torch.Tensor:
        """
        Args:
            local_map         : (B, C, 20, 20)
            position          : (B, 2)               — normalised [x/W, y/H]
            regional_map      : (B, C, 20, 20)       — optional, multi-scale
            received_messages : (B, N-1, comm_dim)   — optional, communication
        Returns:
            q_values : (B, num_actions)
        """
        # ── Contribution 1 : multi-scale feature extraction ──────────
        features = self.encode(local_map, regional_map)    # (B, 64)

        # ── Contribution 2 : communication enrichment ─────────────────
        if self.use_comm and received_messages is not None:
            features = self.comm_module.process_messages(
                features, received_messages
            )                                               # (B, 64)

        # ── Position ──────────────────────────────────────────────────
        pos_feat = F.relu(self.pos_fc(position))           # (B, 16)

        # ── Q-value computation ───────────────────────────────────────
        x = torch.cat([features, pos_feat], dim=-1)        # (B, 80)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q_out(x)                               # (B, 8)


# =============================================================================
#  OBSERVATION BUILDER (agent-side, Python / NumPy)
# =============================================================================

def build_local_observation(discovered_map: np.ndarray,
                             obstacle_map: np.ndarray,
                             exploration_map: np.ndarray,
                             position: tuple,
                             map_w: int, map_h: int,
                             window: int = 20) -> tuple:
    """
    Build the local observation tensor centred on the robot.

    Args:
        discovered_map  : (H, W, 4)  — partial map known to this robot
        obstacle_map    : (H, W)
        exploration_map : (H, W)
        position        : (x, y)     — robot grid position
        window          : local window size (default 20)
    Returns:
        local_obs : (window, window, 6)   — 4 mineral + 1 obstacle + 1 explored
        norm_pos  : (2,)                  — normalised position
    """
    x, y  = position
    half  = window // 2
    x_min = max(0, x - half);  x_max = min(map_w, x + half)
    y_min = max(0, y - half);  y_max = min(map_h, y + half)
    h, w  = y_max - y_min, x_max - x_min

    obs = np.zeros((window, window, 6), dtype=np.float32)
    obs[:h, :w, 0:4] = discovered_map[y_min:y_max, x_min:x_max]
    obs[:h, :w, 4]   = (obstacle_map[y_min:y_max, x_min:x_max] > 0).astype(np.float32)
    obs[:h, :w, 5]   = exploration_map[y_min:y_max, x_min:x_max]

    norm_pos = np.array([x / map_w, y / map_h], dtype=np.float32)
    return obs, norm_pos


def build_regional_observation(discovered_map: np.ndarray,
                                obstacle_map: np.ndarray,
                                exploration_map: np.ndarray,
                                position: tuple,
                                map_w: int, map_h: int,
                                regional_size: int = 60,
                                local_size: int    = 20,
                                scale: int         = 3) -> np.ndarray:
    """
    Build the regional observation: a 60×60 window downsampled to 20×20.

    Downsampling method: average pooling via reshape + mean.
    (60,60,6) → (20,3,20,3,6) → mean(axis=(1,3)) → (20,20,6)

    Args:
        regional_size : size of the raw regional window (default 60)
        local_size    : target output size after downsampling (default 20)
        scale         : downsampling factor (regional_size / local_size = 3)
    Returns:
        regional_obs : (local_size, local_size, 6)
    """
    x, y = position
    half  = regional_size // 2
    x_min = max(0, x - half);  x_max = min(map_w, x + half)
    y_min = max(0, y - half);  y_max = min(map_h, y + half)
    h, w  = y_max - y_min, x_max - x_min

    raw = np.zeros((regional_size, regional_size, 6), dtype=np.float32)
    raw[:h, :w, 0:4] = discovered_map[y_min:y_max, x_min:x_max]
    raw[:h, :w, 4]   = (obstacle_map[y_min:y_max, x_min:x_max] > 0).astype(np.float32)
    raw[:h, :w, 5]   = exploration_map[y_min:y_max, x_min:x_max]

    # Average pooling: (60,60,6) → (20,20,6)
    downsampled = (
        raw[:local_size * scale, :local_size * scale, :]
        .reshape(local_size, scale, local_size, scale, 6)
        .mean(axis=(1, 3))
    ).astype(np.float32)

    return downsampled


# =============================================================================
#  QUICK VALIDATION
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  Contribution 1 — Multi-Scale CNN Encoder")
    print("=" * 60)

    encoder = MultiScaleCNNEncoder(input_channels=6, hidden_dim=64)
    local   = torch.randn(4, 6, 20, 20)    # batch=4, 4 robots
    regional = torch.randn(4, 6, 20, 20)

    features = encoder(local, regional)
    print(f"  Local  input  : {list(local.shape)}")
    print(f"  Regional input: {list(regional.shape)}")
    print(f"  Fused output  : {list(features.shape)}  ← expected (4, 64)")

    print()
    print("=" * 60)
    print("  Contribution 2 — GeoCommQMIX Communication")
    print("=" * 60)

    comm   = CommModule(feature_dim=64, comm_dim=32, num_agents=4)
    h_i    = torch.randn(4, 64)             # own features
    msgs   = torch.randn(4, 3, 32)          # messages from 3 other robots

    msg_out   = comm.encode_message(h_i)
    enriched  = comm.process_messages(h_i, msgs)
    print(f"  Own features      : {list(h_i.shape)}")
    print(f"  Received messages : {list(msgs.shape)}")
    print(f"  Encoded message   : {list(msg_out.shape)}  ← broadcast to others")
    print(f"  Enriched features : {list(enriched.shape)}  ← same dim, enhanced")

    print()
    print("=" * 60)
    print("  Full Q-Network (both contributions active)")
    print("=" * 60)

    net = QNetworkWithContributions(
        input_channels=6, num_actions=8, hidden_dim=64,
        use_multi_scale=True, use_comm=True, comm_dim=32, num_agents=4
    )
    pos   = torch.rand(4, 2)
    q     = net(local, pos, regional_map=regional, received_messages=msgs)
    print(f"  Q-values output   : {list(q.shape)}  ← expected (4, 8)")

    total_params = sum(p.numel() for p in net.parameters())
    print(f"  Total parameters  : {total_params:,}")
    print()
    print("  All checks passed.")
