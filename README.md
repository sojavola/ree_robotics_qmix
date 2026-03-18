# ree_robotics_qmix
# REE Robotics QMIX

> **Exploration coopérative multi-robot de minéraux de terres rares**
> Algorithme QMIX (MARL) intégré dans ROS2 Humble — Stage PRISMALAB

---

## Vue d'ensemble

Quatre robots autonomes explorent une carte géologique de **100 × 100 cellules** pour détecter des gisements de **minéraux de terres rares (REE)**. Le système repose sur le paradigme **CTDE** *(Centralized Training, Decentralized Execution)* : l'entraînement est centralisé sur un nœud trainer, l'exécution est décentralisée sur chaque robot.


                         ┌───────────────────────────────────────────┐
                         │             TRAINER CENTRALISÉ            │
   ┌──────────────┐      │  ┌────────────┐      ┌─────────────────┐  │
   │   Robot 0    │─────►│  │   Replay   │      │   QMIX Network  │  │
   │   Robot 1    │─────►│  │   Buffer   │─────►│  CNN + Mixing   │  │
   │   Robot 2    │─────►│  │ (épisodes) │      │  TD(n=5) loss   │  │
   │   Robot 3    │─────►│  └────────────┘      └────────┬────────┘  │
   └──────┬───────┘      └───────────────────────────────┼───────────┘
          │  ε, poids                                     │
          └───────────────────────────────────────────────┘
          │
   obs. locale (20×20, 6 canaux)    ──►  action (8 directions)


## Packages ROS2

| Package | Rôle |
|---|---|
| `ree_exploration_server` | Génère et publie la carte géologique REE (statique) |
| `ree_exploration_qmix` | Trainer centralisé + 4 agents décentralisés |
| `ree_exploration_viz` | Visualisation RViz2 (carte, robots, minéraux) |

---

## Algorithme — QMIX

### Principe CTDE

QMIX résout le problème de coordination multi-agent en garantissant la **monotonicité** de `Q_tot` par rapport aux `Q_i` individuels. Cela permet à chaque agent de prendre sa décision localement tout en optimisant un objectif global.

```
Q_tot(s, a) = f( Q_1(o_1, a_1), Q_2(o_2, a_2), ..., Q_N(o_N, a_N) )
              ▲
              Monotone → argmax Q_tot = argmax Q_i  (décentralisable)
```

### Architecture du réseau




  Observation locale (20×20×6 canaux)
           │
    ┌──────▼───────────────────────────────┐
    │           CNN Encoder                │
    │  Conv2d(6→32,  k=4, s=2) + BN + ReLU │  →  9×9
    │  Conv2d(32→64, k=3, s=1) + BN + ReLU │  →  7×7
    │  Conv2d(64→64, k=3, s=1) + BN + ReLU │  →  5×5
    │  Flatten → FC(1600→64) + Dropout(0.2)│
    └──────────────────────────────────────┘
           │ features CNN (64)
           │
    Position normalisée (x/W, y/H) → FC(2→16)
           │
    Concat(64+16=80) → FC(80→64) → FC(64→64)
           │
         Q_i(s_i, a_i)   [8 actions par robot]
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
         Q_tot(s, a)   [scalaire]




### Apprentissage — TD(n = 5)

Retour n-pas avec horizon `n = 5` pour réduire le biais sur les récompenses parcimonieuses :

```
G_t^5 = r_t + γ·r_{t+1} + γ²·r_{t+2} + γ³·r_{t+3} + γ⁴·r_{t+4} + γ⁵·Q_target(s_{t+5})

avec γ = 0.995,  Huber loss,  gradient clipping = 10
```

---

## Environnement

| Propriété | Valeur |
|---|---|
| Carte | 100 × 100 cellules |
| Minéraux | 4 types : Oxydes · Silicates · Phosphates · Carbonates |
| Carte | **Statique** — gisements fixes, détection sans consommation |
| Actions | 8 directions (N, S, E, O, NE, NO, SE, SO) |
| Durée épisode | 300 pas par robot |

**Système de récompenses :**

| Événement | Récompense |
|---|---|
| Détection minérale (concentration > 0.3) | `+50` à `+95` |
| Nouvelle cellule visitée | `+1` à `+2` |
| Pas standard | `−0.05` |
| Collision obstacle | `−5.0` |
| Cellule déjà visitée | `−0.5` |

---

## Structure du projet

```
ree_robotics_qmix/
│
├── README.md
│
└── src/
    │
    ├── ree_exploration_server/
    │   └── ree_exploration_server/
    │       ├── server_node.py                  # Publication carte REE
    │       └── advanced_mineral_generator.py   # Génération géologique
    │
    ├── ree_exploration_qmix/
    │   ├── config/
    │   │   └── qmix_params.yaml               # Hyperparamètres
    │   ├── launch/
    │   │   ├── full_system.launch.py          # Lancement complet
    │   │   └── qmix_only.launch.py            # Trainer + agents seuls
    │   └── ree_exploration_qmix/
    │       ├── qmix_trainer_node.py           # Trainer QMIX centralisé
    │       ├── qmix_agent_node.py             # Agent décentralisé
    │       ├── networks.py                    # CNN + Mixing Network
    │       ├── replay_buffer.py               # Buffer épisodique
    │       ├── science_reward_system.py        # Récompenses
    │       └── config.py                      # Configuration dataclass
    │
    └── ree_exploration_viz/
        ├── launch/
        │   └── visualization.launch.py
        └── ree_exploration_viz/
            └── visualization_node.py          # RViz2 MarkerArray

~/.qmix/                                       # Données runtime (auto-créé)
├── models/latest.pt                           # Dernier checkpoint
├── tensorboard/                               # Logs TensorBoard
└── logs/
    ├── episodes.csv                           # 1 ligne / épisode
    ├── training.csv                           # 1 ligne / train step
    └── eval.csv                              # 1 ligne / round eval
```

---

## Prérequis

- Ubuntu 22.04 + ROS2 Humble
- Python 3.10+, PyTorch ≥ 2.0

```bash
pip install torch torchvision tensorboard scipy numpy
```

---

## Installation

```bash
cd /chemin/vers/ree_robotics_qmix/src
colcon build
source install/setup.bash
```

---

## Lancement

### Option A — Système complet (recommandé)

Lance automatiquement dans l'ordre : serveur → trainer + agents → visualisation.

```bash
source install/setup.bash
ros2 launch ree_exploration_qmix full_system.launch.py
```

### Option B — Terminaux séparés

```bash
# Terminal 1 — Serveur REE
source install/setup.bash
ros2 run ree_exploration_server server_node

# Terminal 2 — QMIX (attendre 2s que le serveur publie les cartes)
source install/setup.bash
ros2 launch ree_exploration_qmix qmix_only.launch.py

# Terminal 3 — Visualisation RViz2
source install/setup.bash
ros2 launch ree_exploration_viz visualization.launch.py use_rviz:=false

# Terminal 4 — TensorBoard
tensorboard --logdir ~/.qmix/tensorboard
# → http://localhost:6006
```

---

## Monitoring

### TensorBoard

| Section | Métrique | Description |
|---|---|---|
| `Episode/` | `TotalReward` | Récompense totale brute |
| `Episode/` | `TotalReward_MA10` | Moyenne glissante sur 10 épisodes |
| `Episode/` | `MineralsDetected` | Nombre de détections minérales |
| `Episode/` | `Epsilon` | Valeur ε courante |
| `Robots/` | `Robot{i}_Reward` | Récompense individuelle par robot |
| `Robots/` | `Robot{i}_Minerals` | Minéraux détectés par robot |
| `Train/` | `Loss` | Huber loss TD(5) |
| `Train/` | `GradNorm` | Norme gradient (clippée à 10) |
| `Train/` | `QTot_Mean` | Valeur Q totale moyenne |
| `Train/` | `QTot_Std` | Écart-type Q_tot |
| `Eval/` | `AvgReward` | Récompense évaluation (ε = 0) |
| `CNN/` | `conv1_feature_maps` | Activations visuelles conv1 (32 filtres) |
| `CNN/` | `conv*_weights` | Histogrammes des poids CNN |

### Logs CSV

```bash
tail -20 ~/.qmix/logs/episodes.csv    # Derniers épisodes
cat      ~/.qmix/logs/eval.csv        # Résultats évaluation
tail -20 ~/.qmix/logs/training.csv    # Derniers train steps
```

### Train / Eval split

Tous les **20 épisodes d'entraînement**, un épisode d'évaluation est déclenché :

- `ε = 0.0` — politique greedy pure, sans exploration aléatoire
- L'épisode n'est **pas** ajouté au replay buffer
- Résultats loggés dans `eval.csv` et `Eval/AvgReward` (TensorBoard)

### Continuous Learning

Le trainer sauvegarde automatiquement un checkpoint toutes les **60 secondes** dans `~/.qmix/models/latest.pt`. Au redémarrage, il reprend exactement là où il s'est arrêté (train_step, epsilon, poids réseau, eval_round).

---

## Hyperparamètres (`config/qmix_params.yaml`)

| Paramètre | Valeur | Description |
|---|---|---|
| `gamma` | `0.995` | Facteur de discount |
| `learning_rate` | `0.0001` | Taux d'apprentissage (Adam) |
| `buffer_size` | `5000` | Capacité du replay buffer (épisodes) |
| `batch_size` | `8` | Épisodes par batch |
| `n_steps` | `5` | Horizon retour TD(n) |
| `target_update_freq` | `100` | Fréquence de synchronisation réseau cible |
| `grad_clip` | `10.0` | Clipping de gradient |
| `epsilon_start` | `1.0` | Epsilon initial |
| `epsilon_end` | `0.05` | Epsilon minimal |
| `epsilon_decay` | `20000` | Décroissance epsilon (en steps) |
| `hidden_dim` | `64` | Dimension des couches cachées |
| `map_width / height` | `100` | Taille de la carte |
| `num_robots` | `4` | Nombre d'agents |
| `num_actions` | `8` | Actions disponibles |

---

## Topics ROS2

| Topic | Type | Direction | Description |
|---|---|---|---|
| `/mineral_map` | `Float32MultiArray` | Server → Agents | Carte minérale 100×100×4 |
| `/obstacle_map` | `OccupancyGrid` | Server → Agents | Carte des obstacles |
| `/agent_experience` | `String` (JSON) | Agents → Trainer | Expériences (obs, action, reward) |
| `/qmix/weight_update` | `String` (JSON) | Trainer → Agents | Poids réseau mis à jour |
| `/qmix/epsilon` | `Float32` | Trainer → Agents | Valeur ε courante |
| `/robot_{i}/position` | `Pose2D` | Agent → RViz | Position robot i |
| `/robot_{i}/cmd_vel` | `Twist` | Agent → Sim | Commande de vitesse |

---

## Références

- Rashid et al. (2018) — *QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent RL*
- Mnih et al. (2015) — *Human-level control through deep reinforcement learning* (DQN)
- Peng et al. (2021) — *FACMAC: Factored Multi-Agent Centralised Policy Gradients*


Error loading webview: Error: Could not register service worker: InvalidStateError: Failed to register a ServiceWorker: The document is in an invalid state..
