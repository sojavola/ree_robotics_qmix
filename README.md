# REE Robotics QMIX

> **Cooperative multi-robot exploration of rare earth minerals**
> QMIX algorithm (MARL) integrated into ROS2 Humble — PRISMALAB Internship

---

## Overview

Four autonomous robots explore a **100 × 100 cell** geological map to detect **rare earth mineral (REE)** deposits. The system is based on the **CTDE** paradigm *(Centralized Training, Decentralized Execution)*: training is centralized on a trainer node, execution is decentralized on each robot.


```
                         ┌───────────────────────────────────────────┐
                         │           CENTRALIZED TRAINER             │
   ┌──────────────┐      │  ┌────────────┐      ┌─────────────────┐  │
   │   Robot 0    │─────►│  │   Replay   │      │   QMIX Network  │  │
   │   Robot 1    │─────►│  │   Buffer   │─────►│  CNN + Mixing   │  │
   │   Robot 2    │─────►│  │ (episodes) │      │  TD(n=5) loss   │  │
   │   Robot 3    │─────►│  └────────────┘      └────────┬────────┘  │
   └──────┬───────┘      └───────────────────────────────┼───────────┘
          │  ε, weights                                   │
          └───────────────────────────────────────────────┘
          │
   local obs. (20×20, 6 channels)    ──►  action (8 directions)

```

## ROS2 Packages

| Package | Role |
|---|---|
| `ree_exploration_server` | Generates and publishes the REE geological map (static) |
| `ree_exploration_qmix` | Centralized trainer + 4 decentralized agents |
| `ree_exploration_viz` | RViz2 visualization (map, robots, minerals) |

---

## Algorithm — QMIX

### CTDE Principle

QMIX solves the multi-agent coordination problem by guaranteeing the **monotonicity** of `Q_tot` with respect to individual `Q_i` values. This allows each agent to make its decision locally while optimizing a global objective.

```
Q_tot(s, a) = f( Q_1(o_1, a_1), Q_2(o_2, a_2), ..., Q_N(o_N, a_N) )
              ▲
              Monotone → argmax Q_tot = argmax Q_i  (decentralizable)
```

### Network Architecture

```
  Local observation (20×20×6 channels)
           │
    ┌──────▼───────────────────────────────┐
    │           CNN Encoder                │
    │  Conv2d(6→32,  k=4, s=2) + BN + ReLU │  →  9×9
    │  Conv2d(32→64, k=3, s=1) + BN + ReLU │  →  7×7
    │  Conv2d(64→64, k=3, s=1) + BN + ReLU │  →  5×5
    │  Flatten → FC(1600→64) + Dropout(0.2)│
    └──────────────────────────────────────┘
           │ CNN features (64)
           │
    Normalized position (x/W, y/H) → FC(2→16)
           │
    Concat(64+16=80) → FC(80→64) → FC(64→64)
           │
         Q_i(s_i, a_i)   [8 actions per robot]
           │
    ══════ × 4 robots ══════
           │
    ┌──────▼──────────────────────────────────────┐
    │         Mixing Network (HyperNetwork)        │
    │  W₁ = |HyperNet₁(s)|   B₁ = HyperNet₂(s)   │
    │  W₂ = |HyperNet₃(s)|   B₂ = HyperNet₄(s)   │
    │  Q_tot = ELU(Q · W₁ + B₁) · W₂ + B₂        │
    └──────────────────────────────────────────────┘
           │
         Q_tot(s, a)   [scalar]
```

### Learning — TD(n = 5)

N-step return with horizon `n = 5` to reduce bias on sparse rewards:

```
G_t^5 = r_t + γ·r_{t+1} + γ²·r_{t+2} + γ³·r_{t+3} + γ⁴·r_{t+4} + γ⁵·Q_target(s_{t+5})

with γ = 0.995,  Huber loss,  gradient clipping = 10
```

---

## Environment

| Property | Value |
|---|---|
| Map | 100 × 100 cells |
| Minerals | 4 types: Oxides · Silicates · Phosphates · Carbonates |
| Map | **Static** — fixed deposits, detection without consumption |
| Actions | 8 directions (N, S, E, W, NE, NW, SE, SW) |
| Episode length | 300 steps per robot |

**Reward system:**

| Event | Reward |
|---|---|
| Mineral detection (concentration > 0.3) | `+50` to `+95` |
| New cell visited | `+1` to `+2` |
| Standard step | `−0.05` |
| Obstacle collision | `−5.0` |
| Already visited cell | `−0.5` |

---

## Project Structure

```
ree_robotics_qmix/
│
├── README.md
│
└── src/
    │
    ├── ree_exploration_server/
    │   └── ree_exploration_server/
    │       ├── server_node.py                  # REE map publisher
    │       └── advanced_mineral_generator.py   # Geological generation
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
    │       ├── networks.py                    # CNN + Mixing Network
    │       ├── replay_buffer.py               # Episode buffer
    │       ├── science_reward_system.py        # Reward system
    │       └── config.py                      # Dataclass configuration
    │
    └── ree_exploration_viz/
        ├── launch/
        │   └── visualization.launch.py
        └── ree_exploration_viz/
            └── visualization_node.py          # RViz2 MarkerArray

~/.qmix/                                       # Runtime data (auto-created)
├── models/latest.pt                           # Latest checkpoint
├── tensorboard/                               # TensorBoard logs
└── logs/
    ├── episodes.csv                           # 1 line / episode
    ├── training.csv                           # 1 line / train step
    └── eval.csv                              # 1 line / eval round
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

Automatically launches in order: server → trainer + agents → visualization.

```bash
source install/setup.bash
ros2 launch ree_exploration_qmix full_system.launch.py
```

### Option B — Separate terminals

```bash
# Terminal 1 — REE Server
source install/setup.bash
ros2 run ree_exploration_server server_node

# Terminal 2 — QMIX (wait 2s for server to publish maps)
source install/setup.bash
ros2 launch ree_exploration_qmix qmix_only.launch.py

# Terminal 3 — RViz2 Visualization
source install/setup.bash
ros2 launch ree_exploration_viz visualization.launch.py use_rviz:=false

# Terminal 4 — TensorBoard
tensorboard --logdir ~/.qmix/tensorboard
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
| `Train/` | `QTot_Std` | Q_tot standard deviation |
| `Eval/` | `AvgReward` | Evaluation reward (ε = 0) |
| `CNN/` | `conv1_feature_maps` | Visual activations conv1 (32 filters) |
| `CNN/` | `conv*_weights` | CNN weight histograms |

### CSV Logs

```bash
tail -20 ~/.qmix/logs/episodes.csv    # Latest episodes
cat      ~/.qmix/logs/eval.csv        # Evaluation results
tail -20 ~/.qmix/logs/training.csv    # Latest train steps
```

### Train / Eval split

Every **20 training episodes**, an evaluation episode is triggered:

- `ε = 0.0` — pure greedy policy, no random exploration
- The episode is **not** added to the replay buffer
- Results logged in `eval.csv` and `Eval/AvgReward` (TensorBoard)

### Continuous Learning

The trainer automatically saves a checkpoint every **60 seconds** to `~/.qmix/models/latest.pt`. On restart, it resumes exactly where it left off (train_step, epsilon, network weights, eval_round).

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
| `epsilon_decay` | `20000` | Epsilon decay (in steps) |
| `hidden_dim` | `64` | Hidden layer dimension |
| `map_width / height` | `100` | Map size |
| `num_robots` | `4` | Number of agents |
| `num_actions` | `8` | Available actions |

---

## ROS2 Topics

| Topic | Type | Direction | Description |
|---|---|---|---|
| `/mineral_map` | `Float32MultiArray` | Server → Agents | 100×100×4 mineral map |
| `/obstacle_map` | `OccupancyGrid` | Server → Agents | Obstacle map |
| `/agent_experience` | `String` (JSON) | Agents → Trainer | Experiences (obs, action, reward) |
| `/qmix/weight_update` | `String` (JSON) | Trainer → Agents | Updated network weights |
| `/qmix/epsilon` | `Float32` | Trainer → Agents | Current ε value |
| `/robot_{i}/position` | `Pose2D` | Agent → RViz | Robot i position |
| `/robot_{i}/cmd_vel` | `Twist` | Agent → Sim | Velocity command |

---

## References

- Rashid et al. (2018) — *QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent RL*
- Mnih et al. (2015) — *Human-level control through deep reinforcement learning* (DQN)
- Peng et al. (2021) — *FACMAC: Factored Multi-Agent Centralised Policy Gradients*
