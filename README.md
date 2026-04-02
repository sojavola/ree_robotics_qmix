# REE Robotics QMIX

> **Cooperative multi-robot exploration of rare earth minerals**
> QMIX algorithm (MARL) integrated into ROS2 Humble — PRISMALAB Internship

---

## Overview

Four autonomous robots explore a **100 × 100 cell** geological map to detect **rare earth mineral (REE)** deposits. The system is based on the **CTDE** paradigm *(Centralized Training, Decentralized Execution)*: training is centralized on a trainer node, execution is decentralized on each robot.

This project extends the standard QMIX algorithm with **three original PhD contributions**:

| # | Contribution | Description |
|---|---|---|
| 1 | **Multi-Scale CNN + Spatial Attention** | Local (20×20) + Regional (60×60) views with 25-token cross-scale attention |
| 2 | **GeoCommQMIX** | Learned inter-agent communication with attention gate |
| 3 | **Geo-ICM** | Cooperative intrinsic curiosity module (forward model on CNN features) |

```
                         ┌──────────────────────────────────────────────────┐
                         │             CENTRALIZED TRAINER                  │
   ┌──────────────┐      │  ┌──────────────┐      ┌──────────────────────┐  │
   │   Robot 0    │─────►│  │    Replay    │      │    QMIX Network      │  │
   │   Robot 1    │─────►│  │    Buffer   │─────►│  MultiScale CNN      │  │
   │   Robot 2    │─────►│  │  (episodes) │      │  + GeoComm + ICM    │  │
   │   Robot 3    │─────►│  └──────────────┘      │  TD(n=5) loss        │  │
   └──────┬───────┘      └──────────────────────────────────┬─────────────┘  │
          │  ε, weights                                      │
          └──────────────────────────────────────────────────┘
          │
   local obs. (20×20, 6ch)  +  regional obs. (60×60→20×20)  →  action (8 directions)
```

---

## ROS2 Packages

| Package | Role |
|---|---|
| `ree_exploration_server` | Generates and publishes the REE geological map (static per episode) |
| `ree_exploration_qmix` | Centralized trainer + 4 decentralized agents |
| `ree_exploration_viz` | RViz2 visualization (map, robots, minerals) |

---

## Algorithm — QMIX

### CTDE Principle

QMIX solves the multi-agent coordination problem by guaranteeing the **monotonicity** of `Q_tot` with respect to individual `Q_i` values (IGM — Individual-Global-Max). This allows each agent to make its decision locally while optimizing a global objective.

```
Q_tot(s, a) = f( Q_1(o_1, a_1), Q_2(o_2, a_2), ..., Q_N(o_N, a_N) )
              ▲
              Monotone → argmax Q_tot = argmax Q_i  (decentralizable)
```

Monotonicity is guaranteed by applying `torch.abs()` to hypernetwork output weights.

### Network Architecture

```
  Local observation (20×20×6)        Regional observation (60×60×6 → 20×20×6)
           │                                       │
    ┌──────▼─────────────┐              ┌──────────▼─────────────┐
    │   CNNEncoder (L)   │              │   CNNEncoder (R)        │
    │  Conv(6→32,k=4,s=2)│              │  Conv(6→32,k=4,s=2)    │
    │  Conv(32→64,k=3,s=1│              │  Conv(32→64,k=3,s=1)   │
    │  Conv(64→64,k=3,s=1│              │  Conv(64→64,k=3,s=1)   │
    │  → feature maps    │              │  → feature maps         │
    │    (B, 64, 5, 5)   │              │    (B, 64, 5, 5)        │
    └──────────┬─────────┘              └──────────┬──────────────┘
               │  local_seq (B, 25, 64)             │  regional_seq (B, 25, 64)
               └──────────────┬─────────────────────┘
                              │
                  ┌───────────▼──────────────┐
                  │  Cross-Scale Attention    │
                  │  Q=local, K=V=regional   │
                  │  25 spatial tokens each  │
                  │  → attended (B, 25, 64) │
                  │  + residual + LayerNorm  │
                  └───────────┬──────────────┘
                              │ fusion (64+64=128 → FC → 64)
                              │
                  + position (x/W, y/H → FC → 16)
                  + comm_features (GeoCommQMIX, 64)
                              │
                  ┌───────────▼──────────────┐
                  │   Q-Network Head          │
                  │  FC(144→64) → FC(64→64)  │
                  │  → Q_i(s_i, a_i)  [×8]   │
                  └───────────┬──────────────┘
                              │
                     ══ × 4 robots ══
                              │
                  ┌───────────▼──────────────────────────────┐
                  │         StateEncoder (global state)       │
                  │  Conv2d(6→32, k=8, s=4)  → 24×24         │
                  │  Conv2d(32→64, k=4, s=2) → 11×11         │
                  │  Conv2d(64→64, k=3, s=1) → 9×9           │
                  │  FC(5184 → 256)                           │
                  │  Replaces flat 6×100×100=60k input        │
                  │  Params: 23M → 70,401  (−99.7%)          │
                  └───────────┬──────────────────────────────┘
                              │ state_encoded (256)
                  ┌───────────▼──────────────────────────────┐
                  │    Mixing Network (HyperNetwork)          │
                  │  W₁ = |HyperNet₁(s)|   B₁ = HN₂(s)     │
                  │  W₂ = |HyperNet₃(s)|   B₂ = HN₄(s)     │
                  │  Q_tot = ELU(Q·W₁ + B₁)·W₂ + B₂         │
                  └───────────┬──────────────────────────────┘
                              │
                            Q_tot(s, a)   [scalar]
```

### Contribution 1 — Multi-Scale CNN with Spatial Attention

Each agent receives two observation windows:
- **Local** (20×20×6): high-resolution view of the immediate surroundings
- **Regional** (60×60×6, downsampled to 20×20): wider context at lower resolution

The two CNN encoders produce **25 spatial tokens each** (B, 25, 64) from the 5×5 conv3 feature maps, enabling genuine cross-scale attention (`Q=local, K=V=regional`) rather than a degenerate (B, 1, 64) global average.

**Reference**: Vaswani et al. (2017) — *Attention Is All You Need*

### Contribution 2 — GeoCommQMIX (Learned Communication)

An attention-gated communication module allows agents to share compact messages (64→32 dims):

```
features_i (64)  →  message_i (32)
                         │
                  MultiheadAttention over messages from all N-1 other agents
                         │
                  gate = σ( W · [features_i, attended_msg] )  ∈ [0, 1]
                         │
                  features_i + gate × proj(attended_msg)
```

The gate allows an agent to selectively ignore irrelevant messages. During training, all messages are computed in a single forward pass (CTDE); during execution, messages are exchanged via ROS2 topics.

**References**: Sukhbaatar et al. (2016) — *Learning Multiagent Communication with Backpropagation*;
Das et al. (2019) — *TarMAC: Targeted Multi-Agent Communication*

### Contribution 3 — Geo-ICM (Cooperative Intrinsic Curiosity)

A shared forward model predicts the next CNN features `φ(s_{t+1})` from current features and action:

```
φ(s_t) + a_t  →  FC(72→128→128→64)  →  φ̂(s_{t+1})

r_curiosity = β × || φ̂(s_{t+1}) - φ(s_{t+1}) ||²

r_total = r_extrinsic + β × r_curiosity   (β = 0.1)
```

Curiosity rewards are normalized using a Welford online running mean/variance and clamped to ±3σ. The model is **shared across all agents** (cooperative curiosity): when one agent reduces prediction error in a region, all agents benefit via shared weights.

**References**: Pathak et al. (2017) — *Curiosity-driven Exploration by Self-Supervised Prediction*;
Burda et al. (2019) — *Large-Scale Study of Curiosity-Driven Learning*

### Learning — TD(n = 5)

N-step return with horizon `n = 5` to reduce bias on sparse rewards:

```
G_t^5 = r_t + γ·r_{t+1} + γ²·r_{t+2} + γ³·r_{t+3} + γ⁴·r_{t+4} + γ⁵·Q_target(s_{t+5})

with γ = 0.995,  Huber loss,  gradient clipping = 10
```

Loss is computed with a **single vectorized forward pass** on the full (T × B) batch, instead of T-1 sequential passes.

---

## Environment

| Property | Value |
|---|---|
| Map | 100 × 100 cells |
| Minerals | 4 types: Oxides · Silicates · Phosphates · Carbonates |
| Map | **Static** — fixed deposits per episode, detection without consumption |
| Actions | 8 directions (N, S, E, W, NE, NW, SE, SW) |
| Episode length | 300 steps per robot |
| Episode duration | ~150 seconds (300 × 0.5 s/step) |

**Reward system** — pure mineral signal (no heatmap component):

| Event | Reward |
|---|---|
| Mineral detection (concentration > 0.3) | `+50` to `+110` |
| High concentration bonus (> 0.7) | `+30` additional |
| New cell visited | `+1` |
| Early exploration bonus (step < 50) | `+2` |
| Coverage bonus | `+0.02 × coverage%` |
| Efficiency bonus | `+efficiency × 30` |
| Standard step | `−0.05` |
| Obstacle collision | `−5.0` |
| Already visited cell | `−0.5` |

---

## Project Structure

```
ree_robotics_qmix/
│
├── README.md
├── models/                                # Checkpoints (auto-created)
│   └── qmix/
│       └── latest.pt                     # Latest checkpoint
├── logs/                                  # Training logs (auto-created)
│   ├── qmix/
│   │   ├── episodes.csv
│   │   └── training.csv
│   └── tensorboard/
│
└── src/
    │
    ├── ree_exploration_server/
    │   └── ree_exploration_server/
    │       ├── server_node.py                  # REE map publisher
    │       └── advanced_mineral_generator.py   # Geological generation (elliptic deposits)
    │
    ├── ree_exploration_qmix/
    │   ├── config/
    │   │   └── qmix_params.yaml               # Hyperparameters
    │   ├── launch/
    │   │   ├── full_system.launch.py          # Full system launch
    │   │   └── qmix_only.launch.py            # Trainer + agents only
    │   └── ree_exploration_qmix/
    │       ├── qmix_trainer_node.py           # Centralized QMIX trainer
    │       ├── qmix_agent_node.py             # Decentralized agent
    │       ├── networks.py                    # CNN + Attention + Mixing + StateEncoder
    │       ├── geo_icm.py                     # Geo-ICM forward model (Contribution 3)
    │       ├── replay_buffer.py               # Episode buffer
    │       ├── science_reward_system.py        # Mineral reward system
    │       └── config.py                      # Dataclass configuration
    │
    └── ree_exploration_viz/
        ├── launch/
        │   └── visualization.launch.py
        └── ree_exploration_viz/
            └── visualization_node.py          # RViz2 MarkerArray
```

---

## Prerequisites

- Ubuntu 22.04 + ROS2 Humble
- Python 3.10+, PyTorch ≥ 2.0

```bash
pip install torch torchvision tensorboard scipy numpy
```

---

## Installation

```bash
cd /path/to/ree_robotics_qmix/src
colcon build
source install/setup.bash
```

---

## Launch

### Option A — Full system (recommended)

```bash
source install/setup.bash
ros2 launch ree_exploration_qmix full_system.launch.py
```

### Option B — Separate terminals

```bash
# Terminal 1 — REE Server
source install/setup.bash
ros2 run ree_exploration_server server_node

# Terminal 2 — QMIX trainer + agents
source install/setup.bash
ros2 launch ree_exploration_qmix qmix_only.launch.py

# Terminal 3 — RViz2 Visualization
source install/setup.bash
ros2 launch ree_exploration_viz visualization.launch.py use_rviz:=false

# Terminal 4 — TensorBoard (logs are project-relative)
tensorboard --logdir logs/tensorboard
# → http://localhost:6006
```

---

## Monitoring

### TensorBoard

| Section | Metric | Description |
|---|---|---|
| `Episode/` | `TotalReward` | Raw total reward |
| `Episode/` | `TotalReward_MA10` | Moving average over 10 episodes |
| `Episode/` | `MineralsDetected` | Number of mineral detections |
| `Episode/` | `Epsilon` | Current ε value |
| `Robots/` | `Robot{i}_Reward` | Individual reward per robot |
| `Robots/` | `Robot{i}_Minerals` | Minerals detected per robot |
| `Train/` | `Loss` | Huber loss TD(5) |
| `Train/` | `GradNorm` | Gradient norm (clipped at 10) |
| `Train/` | `QTot_Mean` | Mean total Q value |
| `Train/` | `ICM_Loss` | Geo-ICM forward model loss |
| `Eval/` | `AvgReward` | Evaluation reward (ε = 0) |
| `CNN/` | `conv1_feature_maps` | Visual activations conv1 (32 filters) |

### CSV Logs

```bash
tail -20 logs/qmix/episodes.csv    # Latest episodes
tail -20 logs/qmix/training.csv    # Latest train steps
```

### Train / Eval Split

Every **20 training episodes**, an evaluation episode is triggered:
- `ε = 0.0` — pure greedy policy, no random exploration
- The episode is **not** added to the replay buffer
- Results logged in `Eval/AvgReward` (TensorBoard)

### Continuous Learning

The trainer saves a checkpoint every **60 seconds** to `models/qmix/latest.pt` (project-relative directory). On restart, it resumes exactly where it left off: `train_step`, `epsilon`, network weights, eval round.

---

## Hyperparameters (`config/qmix_params.yaml`)

| Parameter | Value | Description |
|---|---|---|
| `gamma` | `0.995` | Discount factor |
| `learning_rate` | `0.0001` | Learning rate (Adam) |
| `buffer_size` | `5000` | Replay buffer capacity (episodes) |
| `batch_size` | `8` | Episodes per batch |
| `n_steps` | `5` | TD(n) return horizon |
| `target_update_freq` | `100` | Target network sync frequency |
| `grad_clip` | `10.0` | Gradient clipping |
| `epsilon_start` | `1.0` | Initial epsilon |
| `epsilon_end` | `0.05` | Minimum epsilon |
| `epsilon_decay` | `20000` | Epsilon decay in train steps |
| `hidden_dim` | `64` | Hidden layer dimension |
| `curiosity_weight` | `0.1` | Geo-ICM β coefficient |
| `map_width / height` | `100` | Map size |
| `num_robots` | `4` | Number of agents |
| `num_actions` | `8` | Available actions |

---

## ROS2 Topics

| Topic | Type | Direction | Description |
|---|---|---|---|
| `/mineral_map` | `Float32MultiArray` | Server → Agents | 100×100×4 mineral map (published at 2 Hz) |
| `/obstacle_map` | `OccupancyGrid` | Server → Agents | Obstacle map |
| `/underground_layers` | `Float32MultiArray` | Server → Agents | Advanced geological layers |
| `/agent_experience` | `String` (JSON+base64) | Agents → Trainer | Experiences (obs, action, reward) |
| `/qmix/weight_update` | `String` (JSON) | Trainer → Agents | Updated network weights |
| `/qmix/epsilon` | `Float32` | Trainer → Agents | Current ε value |
| `/robot_{i}/position` | `Pose2D` | Agent → RViz | Robot i position |
| `/episode_reset` | `String` | Trainer → Server | Triggers map regeneration |

Large arrays (`mineral_maps`, `regional_maps`) are serialized as **base64-encoded raw bytes** to avoid the ×5–10 size overhead of JSON `.tolist()`.

---

## Engineering Fixes & Optimizations

The following issues were diagnosed and corrected during development:

| # | Issue | Fix | Impact |
|---|---|---|---|
| 1 | Epsilon never decayed | `global_step` (reset each episode) → `train_step` (monotonic) | Policy converges correctly |
| 2 | CNN features drop during inference | Missing `model.eval()` at init + after weight reload | Removes BatchNorm/Dropout in inference |
| 3 | Cross-scale attention degenerate | (B,1,64) global avg → (B,25,64) spatial tokens from conv3 maps | 25 real spatial tokens per view |
| 4 | Hypernetwork: 23M params | `Linear(60000,64)` → `StateEncoder` CNN → 256 dims | 70,401 params (−99.7%) |
| 5 | 750 KB JSON per step | `.tolist()` → `base64.b64encode(arr.tobytes())` | ~5× smaller messages |
| 6 | Reward = 30% noise | Heatmap academic component removed | Pure real mineral signal |
| 7 | Trainer blocks on robot crash | Added `_sync_watchdog()` with 10 s timeout + episode reset | Robust to robot failures |
| 8 | 299 sequential forward passes | Single vectorized pass on (T × B) batch | ~300× faster loss computation |
| 9 | Regional downsampling: 2400-iteration loop | `reshape().mean()` numpy vectorization | ~200× faster preprocessing |
| 10 | `torch.load()` pickle vulnerability | Added `weights_only=True` | Secure checkpoint loading |
| 11 | Infinite loop on start position search | `while True` → `for _ in range(200)` + center fallback | No deadlock at episode start |
| 12 | Models saved to `~/.qmix/` | `__file__`-relative path → `models/qmix/` in project root | Portable across machines |
| 13 | Episode duration: 5 min | `decision_timer` 1.0 s → 0.5 s; `map_timer` 2.0 s → 0.5 s; removed no-op `update_timer` | Episode: 5 min → 2.5 min |

---

## References

- Rashid et al. (2018) — *QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning*. ICML.
- Mnih et al. (2015) — *Human-level control through deep reinforcement learning*. Nature.
- Vaswani et al. (2017) — *Attention Is All You Need*. NeurIPS.
- Pathak et al. (2017) — *Curiosity-driven Exploration by Self-Supervised Prediction*. ICML.
- Burda et al. (2019) — *Large-Scale Study of Curiosity-Driven Learning*. ICLR Workshop.
- Iqbal & Sha (2019) — *Actor-Attention-Critic for Multi-Agent Reinforcement Learning*. ICML.
- Sukhbaatar et al. (2016) — *Learning Multiagent Communication with Backpropagation*. NeurIPS.
- Das et al. (2019) — *TarMAC: Targeted Multi-Agent Communication*. ICML.
- Mnih et al. (2013) — *Playing Atari with Deep Reinforcement Learning* (DQN, CNN encoder). NeurIPS Workshop.
