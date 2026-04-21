# Quadruped Robot Gait Training System

Isaac Gym-based quadruped robot (Unitree Go2) gait control training system using SRBD (Single Rigid Body Dynamics) model and neural network policy for end-to-end training.

## Project Structure

```
single_dog_training/
├── config.py              # Global configuration and environment parameters
├── utils_math.py          # Math utilities (quaternions, rotation matrices, etc.)
├── terrain.py             # Terrain creation (flat/random rough terrain)
├── policy.py              # Neural network policy (MLP)
├── gait.py                # Gait planner (GaitPlanner)
├── srbd.py                # Simplified rigid body dynamics model (SRBDModel)
├── env.py                 # Isaac Gym simulation environment (RealQuadEnv)
├── train.py               # Training main loop
└── play_many_dog.py       # Policy playback script
```

## Features

### Supported Gaits
- **Stand**: All four legs synchronized
- **Trot**: Diagonal gait (FL+RR, FR+RL)
- **Pace**: Lateral gait (FL+RL, FR+RR)
- **Bound**: Front and rear legs synchronized
- **Gallop**: Four legs with sequential phase increments

### Core Technologies
- **SRBD Model**: Simplified single rigid body dynamics for differentiable physics simulation
- **α-Alignment Mechanism**: Blends real physics and SRBD predictions (default α=0.9)
- **Raibert Foothold Planning**: Adaptive foothold calculation based on velocity feedback
- **Multi-Environment Parallel Training**: Supports training multiple robots simultaneously (default 16)
- **GPU Acceleration**: Uses Isaac Gym's GPU physics pipeline

### Loss Functions
Training uses a weighted combination of multiple losses:
- `loss_v`: Velocity tracking (vx, vy)
- `loss_h`: Height maintenance (target 0.35m)
- `loss_omega`: Angular velocity regularization
- `loss_ctrl`: Control input regularization
- `loss_gproj`: Gravity projection (keep body level)
- `loss_foot`: Foot position tracking
- `loss_yaw`: Yaw angle maintenance

## Requirements

### Dependencies
- Python 3.8+
- PyTorch 1.10+
- Isaac Gym Preview 4
- NumPy
- Matplotlib
- tqdm

### Installing Isaac Gym
```bash
# Download Isaac Gym Preview 4
# Extract and enter directory
cd isaacgym/python
pip install -e .
```

⚠️ **Important**: Isaac Gym must be imported before PyTorch. This is already handled in the code.

## Usage

### Training

```bash
# Basic training (1000 iterations, 24 steps each)
python3 train.py

# After training completes, the following files are generated:
# - quad_diffsim_srbd_align_multi_robot.pth  (model weights)
# - quad_diffsim_srbd_align_multi_robot.pt   (TorchScript model for deployment)
# - Various training curves (loss_*.png, vx_curve_*.png, etc.)
```

### Playing Trained Policy

```bash
# Default playback (4 dogs, random velocity commands)
python3 play_many_dog.py

# 16 dogs running together
python3 play_many_dog.py --num_envs 16

# Fixed velocity command (0.5 m/s)
python3 play_many_dog.py --no_rand_cmd

# Specify gait (1=trot)
python3 play_many_dog.py --gait_mode 1

# Specify weight file
python3 play_many_dog.py --weights your_model.pth

# Run for specified steps then stop
python3 play_many_dog.py --max_steps 1000
```

## Configuration

Main configuration in `EnvCfg` class in `config.py`:

### Physics Parameters
- `g = 9.81`: Gravity acceleration
- `h0 = 0.35`: Target height (meters)
- `dt = 0.002`: Simulation timestep (500 Hz)

### Control Parameters
- `action_hold = 5`: Control frequency (100 Hz)
- `pd_kp = 60`: PD controller proportional gain
- `pd_kd = 2`: PD controller derivative gain

### Gait Parameters
- `step_freq = 1.6`: Step frequency (Hz)
- `swing_height = 0.12`: Swing height (meters)
- `gait_mode = 1`: Gait mode (-1=random, 0=stand, 1=trot, 2=pace, 3=bound, 4=gallop)

### Training Parameters
- `num_envs = 16`: Number of parallel environments
- `alpha_align = 0.9`: SRBD alignment coefficient
- `train_no_aerial = True`: Disable aerial phase during training (warm-up)

### Global Switches
- `PURE_PAPER_MODE = True`: Pure paper version (no engineering tricks)
- `USE_COMPLEX_TERRAIN = False`: Use random rough terrain
- `ONLY_ITERATE_NO_RESET = True`: Only reset on first iteration

## Training Output

After training completes, the following files are generated:

### Model Files
- `quad_diffsim_srbd_align_multi_robot.pth`: PyTorch model weights
- `quad_diffsim_srbd_align_multi_robot.pt`: TorchScript model (for ROS2 deployment)

### Training Curves
- `loss_curve_srbd_align.png`: Total loss curve
- `loss_components_curve_srbd_align.png`: Individual loss components
- `vx_curve_srbd_align.png`: Body forward velocity curve
- `vx_curve_srbd_align_smooth.png`: Smoothed velocity curve
- `reward_curve_srbd_align.png`: Reward curve

### Data Files
- `*.npy`: NumPy arrays of various metrics (for post-analysis)

## Code Architecture

### Module Responsibilities

- **config.py**: Centralized management of all configuration parameters and global switches
- **utils_math.py**: Provides quaternion, rotation matrix, gravity projection and other math utilities
- **terrain.py**: Creates flat or random rough terrain
- **policy.py**: Defines neural network policy (36-dim input → 256×256 → 12-dim output)
- **gait.py**: `GaitPlanner` class, handles gait planning, phase management, foothold calculation
- **srbd.py**: `SRBDModel` class, implements simplified rigid body dynamics forward propagation
- **env.py**: `RealQuadEnv` class, wraps Isaac Gym simulation environment
- **train.py**: Training main loop, includes loss calculation, backpropagation, model saving

### Key Design Patterns

#### GaitPlanner (Gait Planner)
Uses `__getattr__` and `__setattr__` to proxy access to environment attributes, avoiding circular dependencies:
```python
self.gait = GaitPlanner(self)
pref, stance_mask = self.gait._update_foot_targets_from_command(phases, p_foot)
```

#### SRBDModel (Simplified Rigid Body Dynamics)
Also uses proxy pattern to implement differentiable physics propagation:
```python
self.srbd = SRBDModel(self)
self.srbd._srbd_step(f_world=f_est, q_ref12=qref, dt=dt)
```

#### α-Alignment Mechanism
In each training step, blends Isaac Gym's real physics with SRBD predictions:
```python
env.srbd_p = env.base_pos + alpha * (env.srbd_p - env.srbd_p.detach())
env.srbd_v = env.base_lin + alpha * (env.srbd_v - env.srbd_v.detach())
```

## Debugging Features

### Initial Fall Debugging Prints
Set in `config.py`:
```python
DBG_INIT_FALL = True
DBG_INIT_FALL_STEPS = 250  # Only print first 250 steps
DBG_INIT_FALL_EVERY = 5    # Print every 5 steps
DBG_INIT_FALL_ENV = 0      # Only watch dog 0
```

### Gradient Flow Analysis
Training automatically prints gradient norms for each layer to diagnose vanishing/exploding gradients.

### Direction Checking
Prints body velocity direction every 20 iterations to verify velocity tracking is correct.

## FAQ

### Q: Robot keeps flipping forward during training?
A: Check the following:
1. `train_no_aerial = True` (disable aerial phase)
2. `alpha_align = 0.9` (sufficient alignment coefficient)
3. Step frequency not too high (recommend 1.5-1.8 Hz)
4. Check if `last_contact_z` is updated correctly

### Q: Isaac Gym import fails?
A: Ensure Isaac Gym is imported before PyTorch. This is already handled in the code, but if issues persist, check:
```python
# Correct order
from isaacgym import gymapi
import torch

# Wrong order
import torch
from isaacgym import gymapi  # Will error
```

### Q: Training is slow?
A:
1. Ensure `use_gpu_pipeline = True`
2. Increase `num_envs` (more parallel environments)
3. Turn off `use_viewer` (don't render during training)
4. Use a faster GPU

### Q: How to deploy to real robot?
A: After training, use the TorchScript model:
```python
model = torch.jit.load("quad_diffsim_srbd_align_multi_robot.pt")
action = model(observation)  # (1, 36) -> (1, 12)
```

## Citation

If you use this code, please cite the relevant paper (fill in according to your actual paper):

```bibtex
@article{your_paper,
  title={Your Paper Title},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## License

(Fill in according to your project license)

## Contact

(Fill in according to your contact information)

---

**Note**: This project is based on Isaac Gym Preview 4 and is for research and educational purposes only.
