# rl_agent_ppo — PPO Lane Following pour Duckiebot + Gazebo Harmonic

Implémentation PPO **from scratch en PyTorch** (sans Stable-Baselines3), avec un espace d'actions **continu**.
C'est la version décrite dans le rapport (chapitre 2.7 et section 3.3.3 — piste circulaire, 1 000 000 pas).

## Architecture

```
rl_agent_ppo/
├── rl_agent_ppo/
│   ├── ppo_model.py          ← ActorCritic (backbone Tanh partagé) + RolloutBuffer + PPOAgent
│   ├── ppo_reward.py         ← Récompense PPO (échelle [-5, +0.30] par step)
│   ├── ppo_training_node.py  ← Nœud ROS2 d'entraînement (10 Hz)
│   ├── lane_detector.py      ← Détection de la piste par caméra (OpenCV)
│   └── diag_node.py          ← Nœud de diagnostic de la détection
├── launch/ppo_train.launch.py
├── config/bridge.yaml
└── test/test_lane_detector.py
```

## Principe

| Élément | Valeur |
|---|---|
| État | `[d, θ, ω/2]` (écart latéral, erreur d'angle, vitesse angulaire `/odom`) |
| Actions (continues) | `v ∈ [0.05, 0.25] m/s`, `ω ∈ [-1.0, 1.0] rad/s` — Gaussienne, `σ ∈ [1e-4, 1]` |
| Récompense | `-5.0` hors piste · `-0.8` pas de bordure détectée · `0.10·e^(-5d²)+0.05` une bordure · `0.30·e^(-5d²)·e^(-2.5θ²)` deux bordures |
| Fin d'épisode | reward cumulé ≤ −250 ou 2000 steps sur la piste |
| Reset | 8 positions de départ (tous les 45°) ± 0.05 m |

## Hyperparamètres

| Paramètre | Valeur |
|---|---|
| Learning rate (Adam, eps=1e-5) | 3e-4 |
| γ / λ (GAE) | 0.99 / 0.95 |
| Clip ε | 0.2 |
| c_v / c_H | 0.5 / 0.01 |
| Rollout T | 256 steps |
| Epochs K / mini-batch | 4 / 64 |
| Gradient clip | 0.5 |

## Lancer l'entraînement

```bash
# Terminal 1 : Gazebo + robot
ros2 launch duckiebot_description gazebo.launch.py

# Terminal 2 : bridge + entraînement PPO
ros2 launch rl_agent_ppo ppo_train.launch.py
```

Sorties :
- modèle : `~/ros/models/ppo_model.pt` (sauvegardé tous les 1000 steps)
- logs : `~/ros/logs/ppo_steps.csv`, `~/ros/logs/ppo_episodes.csv`
