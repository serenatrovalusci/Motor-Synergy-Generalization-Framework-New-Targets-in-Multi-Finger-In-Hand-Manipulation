# Motor Synergy Generalization Framework: New Targets in Multi-Finger In-Hand Manipulation

A A reinforcement learning framework for dexterous manipulation that transfers motor synergies extracted from a source task to accelerate learning on new, unseen target objects. A SAC+HER agent is first trained on block rotation to mastery; its joint-action trajectories are factorised via PCA into a compact synergy basis; target-task agents then act in this low-dimensional latent space, achieving faster convergence and higher sample efficiency than a full action-space baseline.

> **Author:** Serena Trovalusci  
> **Supervisor:** Alessandro De Luca · **Co-supervisor:** Mitsuhiro Hayashibe  
> **Institution:** Dipartimento di Ingegneria Informatica, Automatica e Gestionale (DIAG), Sapienza University of Rome  
> **Thesis:** [`report&slides/Tesi.pdf`](report&slides/Tesi.pdf)  
> **Slides:** [`report&slides/defense_presentation.pdf`](report&slides/defense_presentation.pdf)

---

## What is this project?

Dexterous in-hand manipulation is one of the hardest problems in robotics — the Shadow Hand has 20 degrees of freedom and must rotate objects to arbitrary goal orientations. Standard RL approaches require millions of environment steps per object.

This project exploits **motor synergies**: the observation that coordinated finger movements lie on a low-dimensional manifold. By extracting a K-dimensional basis from source-task trajectories and using it as a fixed decoder, target-task agents search a K-dimensional space (K=5) instead of the native 20-dimensional one.

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Pipeline Overview                            │
│                                                                      │
│  Stage A — Source task                                               │
│  SAC+HER trains on block rotation (full 20-DoF action space)         │
│       │                                                              │
│       ▼                                                              │
│  Collect joint-action trajectories  →  trajectory.npz               │
│       │                                                              │
│       ▼                                                              │
│  Stage B — Synergy extraction                                        │
│  PCA fits a K-dim basis W ∈ R^{K×20}  →  pca/block_K5              │
│       │                                                              │
│       ▼                                                              │
│  Stage C — Target tasks                                              │
│  SAC+HER acts in R^K  →  W decodes to R^20  →  env.step()           │
│  Trained on: cylinder · egg · large_cube · small_cube · sphere       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Environment

All experiments use the **Shadow Dexterous Hand** with continuous touch sensors from [Gymnasium-Robotics](https://robotics.farama.org/):

| Task | Object | Environment ID |
|------|--------|---------------|
| Source | Standard cube (2.5 cm) | `HandManipulateBlockRotateXYZ_ContinuousTouchSensors-v1` |
| Target | Large cube (3.0 cm) | `HandManipulateBlockRotateXYZ_ContinuousTouchSensors-v1` |
| Target | Small cube (2.0 cm) | `HandManipulateBlockRotateXYZ_ContinuousTouchSensors-v1` |
| Target | Egg | `HandManipulateEggRotateXYZ_ContinuousTouchSensors-v1` |
| Target | Cylinder | `HandManipulateBlockRotateXYZ_ContinuousTouchSensors-v1` |
| Target | Sphere | `HandManipulateBlockRotateXYZ_ContinuousTouchSensors-v1` |

> All target objects (except for the egg) share the same base environment. Object geometry and dimensions are defined via custom MuJoCo XML files — the cube size variants change block dimensions, while the egg, cylinder, and sphere replace the object mesh entirely.

The hand has **20 joints** (M=20), each controlled by a continuous torque command. The synergy basis reduces this to **K=5 latent dimensions**.

---

## Repository Structure

```
.
├── sac_her_pipeline.py          # Main script: train / eval / collect
├── synergy.py                   # SpatialSynergy class (PCA / NMF)
├── synergy_extract_analyze.py   # Offline synergy extraction + diagnostic plots
├── synergy_replay.py            # Replay saved trajectories in the simulator
├── plot.py                      # Final comparison plots (success & efficiency)
├── trajectory.npz               # Source-task trajectories (block, 6M timesteps)
├── requirements.txt
├── pca/
│   ├── block_K4                 # Pre-fitted synergy model, K=4
│   ├── block_K5                 # Pre-fitted synergy model, K=5  ← used by default
│   └── block_K6                 # Pre-fitted synergy model, K=6
├── models/
│   ├── synergy_K5/              # Synergy-constrained agents (K=5)
│   │   ├── cylinder/
│   │   │   ├── best_model.zip
│   │   │   └── vecnorm_best.pkl
│   │   ├── egg/
│   │   ├── large_cube/
│   │   ├── small_cube/
│   │   └── sphere/
│   └── full_action/             # Full action-space baseline agents
│       ├── cylinder/
│       ├── egg/
│       ├── large_cube/
│       ├── small_cube/
│       └── sphere/
├── plots/                       # Pre-generated result figures
│   ├── comparison_*_success.png
│   └── comparison_*_efficiency.png
└── report&slides/
    ├── Tesi.pdf
    └── defense_presentation.pdf
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/serenatrovalusci/Motor-Synergy-Generalization-Framework-New-Targets-in-Multi-Finger-In-Hand-Manipulation.git
cd Motor-Synergy-Generalization-Framework-New-Targets-in-Multi-Finger-In-Hand-Manipulation
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> Requires Python ≥ 3.10 and a MuJoCo-compatible system. GPU optional but recommended for training.

### 3. Run evaluation (pre-trained weights included)

```bash
# Synergy agent — egg
python sac_her_pipeline.py --task eval \
    --env-id HandManipulateEggRotate_ContinuousTouchSensors-v1 \
    --synergy-path pca/block_K5 \
    --save-dir models/synergy_K5/egg \
    --eval-episodes 100

# Full action-space baseline — egg
python sac_her_pipeline.py --task eval \
    --full-action-space \
    --env-id HandManipulateEggRotate_ContinuousTouchSensors-v1 \
    --save-dir models/full_action/egg \
    --eval-episodes 100
```

---

## Pipeline — Step by Step

### Stage A — Train source task (full action space)

```bash
python sac_her_pipeline.py --task train \
    --full-action-space \
    --env-id HandManipulateBlockRotateXYZ_ContinuousTouchSensors-v1 \
    --save-dir runs/block_full \
    --timesteps 6000000
```

### Collect source-task trajectories

```bash
python sac_her_pipeline.py --task collect \
    --full-action-space \
    --env-id HandManipulateBlockRotateXYZ_ContinuousTouchSensors-v1 \
    --save-dir runs/block_full \
    --traj-episodes 200 \
    --traj-save-path trajectory.npz
```

> The pre-collected `trajectory.npz` (200 episodes, 6M training timesteps) is already included.

### Stage B — Extract synergies

```bash
python synergy_extract_analyze.py \
    --npz-path trajectory.npz \
    --method pca \
    --n-synergies 5 \
    --out-path pca/block_K5 \
    --k-list "1,2,3,4,5,6,8,10,15,20"
```

> Pre-fitted models for K=4,5,6 are already included in `pca/`.

### Stage C — Train target tasks (synergy space)

```bash
python sac_her_pipeline.py --task train \
    --env-id HandManipulateEggRotate_ContinuousTouchSensors-v1 \
    --synergy-path pca/block_K5 \
    --save-dir runs/synergy_K5_egg \
    --timesteps 2000000
```

### Monitor training

```bash
tensorboard --logdir runs/synergy_K5_egg/tb_logs
```

### Generate comparison plots

```bash
python plot.py
```

---

## Results

Pre-trained models for all 5 target objects are included under `models/` for both the synergy (K=5) and full action-space configurations. Pre-generated plots are in `plots/`.

The synergy-constrained agent consistently reaches comparable success rates with significantly fewer environment steps and shorter wall-clock time across all target objects, confirming that the block-derived synergy basis transfers effectively regardless of object geometry.

| Target Object | Steps — Full 20-DoF | Steps — Synergy 5-DoF | Time — Full (h) | Time — Synergy (h) | Speedup |
|--------------|--------------------|-----------------------|-----------------|-------------------|---------|
| Large Cube   | 3.0 M              | 2.0 M                 | 5.65            | 3.82              | ~1.5×   |
| Small Cube   | 2.6 M              | 1.5 M                 | 7.57            | 4.12              | ~1.8×   |
| Egg          | 2.3 M              | 1.3 M                 | 6.97            | 3.78              | ~1.8×   |
| Cylinder     | 1.5 M              | 0.7 M                 | 4.38            | 1.90              | ~2.3×   |
| **Sphere**   | **1.2 M**          | **0.2 M**             | **3.37**        | **0.53**          | **~6.4×** |

---

## Demo — Sphere at 500k Timesteps

The videos below show both agents on the **Sphere** task after the same number of training steps (500k), making the comparison direct and fair. At this checkpoint, the synergy agent has already converged (it reaches full performance at ~200k steps), while the full action-space agent is still learning (it requires ~1.2M steps to converge.

**Full action space** *(500k steps — still learning)*

https://github.com/user-attachments/assets/51d9ec83-a407-4bfe-8e73-728304b20445

**Synergy K=5** *(500k steps — already converged)*

https://github.com/user-attachments/assets/1e61dc22-dd75-4ede-b01f-aecb41ed33f6

---

## Key Techniques

- **SAC + HER** (Soft Actor-Critic + Hindsight Experience Replay) for goal-conditioned dexterous manipulation
- **PCA synergy extraction** from source-task joint-action trajectories
- **Low-dimensional action space** via a frozen linear decoder W ∈ R^{K×20}
- **VecNormalize** with deep-copied obs_rms to prevent eval statistics from corrupting training
- **SaveVecNormalizeOnBest** callback to keep `best_model.zip` and `vecnorm_best.pkl` in sync

---

## Reference

This work extends the motor synergy generalization framework introduced in:

> Kutsuzawa & Hayashibe, *Motor synergy generalization framework for new targets in multi-planar and multi-directional reaching task*, Royal Society Open Science, 2022. [https://doi.org/10.1098/rsos.211721](https://doi.org/10.1098/rsos.211721)
