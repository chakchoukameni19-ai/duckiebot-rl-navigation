# rl_agent_dqn — DQN Lane Following pour Duckiebot + Gazebo Harmonic

## Architecture

```
rl_agent_dqn/
├── rl_agent_dqn/
│   ├── __init__.py
│   ├── dqn_model.py          ← Réseau DQN + Replay Buffer + Agent
│   ├── lane_detector.py      ← Détection piste par vision (OpenCV)
│   ├── reward.py             ← Fonction de récompense
│   ├── dqn_training_node.py  ← Nœud ROS2 entraînement
│   └── dqn_eval_node.py      ← Nœud ROS2 évaluation (ε=0)
├── launch/
│   ├── dqn_train.launch.py   ← Lance Gazebo + DQN training
│   └── dqn_eval.launch.py    ← Lance uniquement l'évaluation
├── test/
│   └── test_lane_detector.py ← Test standalone sans ROS
├── config/
│   └── bridge.yaml           ← Référence pour bridge Gazebo↔ROS
├── setup.py
└── package.xml
```

## Principe

### État (dimension 3)
| Variable | Description | Plage |
|----------|-------------|-------|
| `d`      | Distance latérale au centre de la piste | [-1, 1] |
| `theta`  | Angle relatif robot↔piste | [-1, 1] |
| `omega`  | Vitesse angulaire normalisée | [-1, 1] |

### Actions (5 discrètes)
| Index | Description | lin (m/s) | ang (rad/s) |
|-------|-------------|-----------|-------------|
| 0 | Tout droit | 0.20 | 0.00 |
| 1 | Virage gauche doux | 0.15 | +0.40 |
| 2 | Virage droite doux | 0.15 | -0.40 |
| 3 | Virage gauche fort | 0.10 | +0.80 |
| 4 | Virage droite fort | 0.10 | -0.80 |

### Récompense
```
r = exp(-4·d²) × exp(-2·θ²) + 0.1    si piste détectée et |d| ≤ 0.9
r = -5.0                               si |d| > 0.9  (hors piste)
r = -0.5                               si piste non détectée
```

### Hyperparamètres DQN
| Paramètre | Valeur |
|-----------|--------|
| Couches cachées | 128-128 |
| Optimizer | Adam lr=1e-3 |
| γ (discount) | 0.95 |
| ε : 1.0 → 0.05 | decay=2000 steps |
| Batch size | 64 |
| Buffer | 50 000 |
| Target sync | 200 steps |
| Loss | Huber (SmoothL1) |

## Installation

```bash
# Depuis la racine du workspace ROS2
cd ~/ros2_ws/src
# Copier le dossier rl_agent_dqn ici
cd ~/ros2_ws
pip3 install torch opencv-python cv-bridge
colcon build --packages-select rl_agent_dqn
source install/setup.bash
```

## Lancer l'entraînement

```bash
# Entraînement complet (Gazebo + DQN)
ros2 launch rl_agent_dqn dqn_train.launch.py

# Reprendre un entraînement existant
ros2 launch rl_agent_dqn dqn_train.launch.py load_model:=true

# Mode évaluation pure (Gazebo déjà lancé)
ros2 launch rl_agent_dqn dqn_eval.launch.py
```

## Vérifier le bridge caméra

Le topic image Gazebo est bridgé via `ros_gz_image` :
```bash
ros2 topic list | grep camera
# Doit afficher : /camera/image_raw
```

## Test sans ROS

```bash
cd ~/ros2_ws/src/rl_agent_dqn
python3 test/test_lane_detector.py
```

## Logs pendant l'entraînement

```
ep=  12 | step=  1200 | ε=0.549 | loss=0.0234 | R_ep=47.31 | d=0.023
```

- **ep** : numéro d'épisode
- **ε** : taux d'exploration actuel
- **R_ep** : récompense cumulée de l'épisode
- **d** : distance latérale courante (idéalement → 0)

## Ajustements si nécessaire

### Détection caméra trop imprécise
Modifier dans `lane_detector.py` :
```python
WHITE_LOW  = np.array([0,   0, 180])   # ↑ 180 si sur-détection
TRACK_WIDTH_PX = 200                   # ajuster selon zoom caméra
```

### Convergence lente
Modifier dans `dqn_model.py` :
```python
eps_decay = 1000    # ↓ accélère la convergence (moins d'exploration)
lr        = 2e-3    # ↑ apprentissage plus rapide
```

### Sorties de piste fréquentes
Modifier dans `reward.py` :
```python
OUT_PENALTY = -10.0   # ↑ pénalise davantage les sorties
K_D         = 6.0     # ↑ pénalise davantage l'écart latéral
```
