# Duckiebot — Navigation autonome par Deep Reinforcement Learning (DQN & PPO)

Projet de Fin d'Année (PFA) 2026 — ENET'COM Sfax — **Mariem Ameni**

Un Duckiebot apprend à suivre une piste circulaire dans **Gazebo Harmonic** sous **ROS 2 Jazzy**, à partir
d'un état compact `s = [d, θ, ω]` (écart latéral et angle issus de la caméra, vitesse angulaire issue de `/odom`).
Deux algorithmes sont implémentés et comparés :

| | **DQN** (`rl_agent_dqn`) | **PPO** (`rl_agent_ppo`) |
|---|---|---|
| Type | Value-based, off-policy | Policy gradient, on-policy (Actor-Critic) |
| Actions | 5 discrètes (tout droit, ±0.40, ±0.80 rad/s) | Continues : `v ∈ [0.05, 0.25]`, `ω ∈ [-1, 1]` |
| Mémoire | Replay buffer 50 000 | Rollout buffer 256 |
| Entraînement | 1 000 épisodes, départ fixe | 1 000 000 pas, 8 positions de départ |

Le rapport complet est disponible dans [`docs/Rapport_PFA_Mariem_Ameni_2026.pdf`](docs/Rapport_PFA_Mariem_Ameni_2026.pdf).

## Structure du dépôt

```
duckiebot-rl-navigation/
├── src/                          ← workspace ROS 2 (colcon)
│   ├── duckiebot_description/    ← modèle URDF/Xacro, meshes, monde Gazebo, bridge ROS↔Gazebo
│   ├── rl_agent_dqn/             ← agent DQN (entraînement, évaluation, variante odométrie)
│   ├── rl_agent_ppo/             ← agent PPO (actions continues)
│   ├── tf_tools/                 ← publication du TF odom → base_link
│   └── duckietown_msgs/          ← messages Duckietown (pour le bridge ROS 1 ↔ ROS 2)
├── models/
│   ├── dqn/dqn_model.pt          ← modèle DQN entraîné
│   ├── ppo/ppo_model.pt          ← modèle PPO entraîné
│   └── ppo_straight_line_sb3/    ← modèle PPO Stable-Baselines3 (tâche ligne droite / robot réel)
├── results/                      ← logs CSV d'entraînement utilisés pour les courbes du rapport
│   ├── dqn/
│   ├── ppo/
│   └── ppo_straight_line/
├── scripts/                      ← nœuds de rejeu des CSV vers /cmd_vel (démonstration)
└── docs/                         ← rapport PFA, arbre TF
```

## Prérequis

- Ubuntu 24.04, ROS 2 Jazzy, Gazebo Harmonic (`ros_gz_bridge`, `ros_gz_image`, `ros_gz_sim`)
- Python 3.12 : `torch`, `numpy`, `opencv-python`, `cv_bridge`

```bash
sudo apt install ros-jazzy-ros-gz ros-jazzy-cv-bridge ros-jazzy-xacro
pip install torch numpy opencv-python
```

## Compilation

```bash
git clone <url-du-depot> ~/ros
cd ~/ros
colcon build --symlink-install
source install/setup.bash
```

> Pendant l'entraînement, les nœuds sauvegardent les modèles dans `~/ros/models/` et les logs dans `~/ros/logs/`.
> Les modèles fournis dans ce dépôt sont rangés dans `models/dqn/` et `models/ppo/`.

## Utilisation

### DQN
```bash
ros2 launch rl_agent_dqn dqn_train.launch.py                  # Gazebo + entraînement
ros2 launch rl_agent_dqn dqn_train.launch.py load_model:=true # reprise
ros2 launch rl_agent_dqn dqn_eval.launch.py model_path:=$HOME/ros/models/dqn/dqn_model.pt
```
Détails : [`src/rl_agent_dqn/README.md`](src/rl_agent_dqn/README.md)

### PPO
```bash
ros2 launch duckiebot_description gazebo.launch.py   # terminal 1
ros2 launch rl_agent_ppo ppo_train.launch.py         # terminal 2
```
Détails : [`src/rl_agent_ppo/README.md`](src/rl_agent_ppo/README.md)

### Diagnostic de la détection de piste
```bash
ros2 run rl_agent_dqn diag_node
```

## Déploiement sur le robot réel (sim-to-real)

Le Duckiebot tourne sous ROS 1 ; le PC de contrôle sous ROS 2 Jazzy. La communication passe par
[`ros1_bridge`](https://github.com/ros2/ros1_bridge) (`dynamic_bridge`) exécuté dans un conteneur Docker
ROS 1 Noetic + ROS 2 Foxy. Le paquet `ros1_bridge` n'est pas inclus dans ce dépôt : voir le dépôt officiel.

## Résultats principaux

Voir le chapitre 3 du rapport. Les CSV bruts sont dans `results/`.
