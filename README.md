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
├── play_many_dog.py       # Policy playback script
├── setup.py               # Build script for the SRBD CUDA extension
├── test_srbd_kernel.py    # Forward/backward parity test for the CUDA kernel
└── src/
    ├── srbd_ext.cpp       # pybind11 bindings for the CUDA kernel
    └── srbd_cuda.cu       # Fused CUDA kernel: foot FK + SRBD dynamics step
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
- **Multi-Environment Parallel Training**: Supports training multiple robots simultaneously (default 16, scales to ~1000+)
- **GPU Acceleration**: Uses Isaac Gym's GPU physics pipeline
- **Custom CUDA Kernel for SRBD** (optional): Fused kernel that replaces the per-step PyTorch ops with a single launch; toggled via `CUDA_KERNEL_SRBD` in `config.py`. Backward compatibility with autograd is preserved (see [SRBD CUDA Kernel](#srbd-cuda-kernel-optional))

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
- PyTorch 1.10+ (built with CUDA support)
- Isaac Gym Preview 4
- NumPy
- Matplotlib
- tqdm

### Optional (only for the custom SRBD CUDA kernel)
- CUDA Toolkit matching your PyTorch build (`nvcc --version` must work)
- A C++ toolchain (gcc/clang on Linux, MSVC on Windows) — the same one PyTorch was built against

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
- `CUDA_KERNEL_SRBD = False`: Use the custom fused CUDA kernel for `_srbd_step`. `False` = pure PyTorch (default, always works). `True` = CUDA kernel (requires `python setup.py build_ext --inplace` first; see [SRBD CUDA Kernel](#srbd-cuda-kernel-optional)).

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

## SRBD CUDA Kernel (optional)

The per-step centroidal dynamics in `SRBDModel._srbd_step` can run through either of two backends, selected by a single switch in `config.py`:

```python
# config.py
CUDA_KERNEL_SRBD = False   # PyTorch (default — always works, no build step)
CUDA_KERNEL_SRBD = True    # Custom fused CUDA kernel (requires building the extension)
```

When `True`, `srbd.py` imports the compiled `srbd_cuda_ext` module and dispatches both directions through a `torch.autograd.Function` wrapper (`SRBDStepFunction`):
- **Forward** is a single fused kernel in `src/srbd_cuda.cu` (foot FK + force/torque accumulation + Newton-Euler dynamics + quaternion integration), one thread per environment.
- **Backward** is a hand-written analytic adjoint kernel in the same file — no PyTorch replay, no `torch.autograd.grad` re-execution. It recomputes the forward intermediates and applies the chain rule directly, returning gradients for all six tensor inputs (`p, v, q, w, f_world, q_ref12`).

`loss.backward()` works identically under both backends. Gradients propagate through the SRBD state across the full rollout, matching the pure-PyTorch behavior. The kernel is built with `--use_fast_math`, so gradients agree with the PyTorch path to ≈ 1e-4 absolute (1-2 ULP per `sinf`/`cosf`/`rsqrtf` call, accumulated through the chain).

If the extension is not built, the code prints a warning and silently falls back to the PyTorch path, so toggling the flag is always safe.

### Building the extension

The extension uses `torch.utils.cpp_extension.CUDAExtension`. By default `setup.py` cross-compiles for every major NVIDIA architecture from Pascal (sm_60) through Hopper (sm_90), plus PTX for forward-compatibility with future GPUs (≈ 3–8 minutes the first time):

```bash
# From the project root, with your conda environment active
python setup.py build_ext --inplace
```

This produces `srbd_cuda_ext*.so` (Linux) / `srbd_cuda_ext*.pyd` (Windows) in the project root.

For faster iteration during development, restrict the build to the GPU on the current machine (~30 s):

```bash
# Linux
TORCH_CUDA_ARCH_LIST="native" python setup.py build_ext --inplace
```

```powershell
# Windows PowerShell
$env:TORCH_CUDA_ARCH_LIST = "native"
python setup.py build_ext --inplace
```

### Running training with the CUDA kernel

```bash
# 1) Set the flag in config.py:
#    CUDA_KERNEL_SRBD = True
# 2) Run training as usual:
python train.py
```

On the first import you should see:

```
[SRBD] Custom CUDA kernel active (CUDA_KERNEL_SRBD=True in config.py).
```

If the extension is missing or fails to import, you will instead see:

```
[SRBD] WARNING: CUDA_KERNEL_SRBD=True but extension not found (...).
[SRBD]          Falling back to PyTorch implementation.
[SRBD]          Run: python setup.py build_ext --inplace
```

### Testing the kernel

After every rebuild, run the parity test to verify the CUDA kernel matches the PyTorch reference path on the same inputs:

```bash
python test_srbd_kernel.py
```

The test runs three checks and exits with code 0 on success:

1. **Forward parity** — compares `srbd_step_forward` (CUDA) against an inline copy of the inline PyTorch path from `srbd.py:_srbd_step` on a random batch (B = 16). Tolerance `atol=1e-4, rtol=1e-3`.
2. **Backward parity** — builds random upstream gradients, runs `torch.autograd.backward` on the PyTorch reference, calls `srbd_step_backward` (CUDA) directly, and compares all six input gradients (`p, v, q, w, f_world, q_ref12`). Tolerance `atol=1e-3, rtol=1e-2`.
3. **Autograd wrapper round-trip** — calls `SRBDStepFunction.apply(...)` end-to-end, runs `.backward()` on a weighted sum of outputs, and compares `.grad` of each input against the PyTorch reference. This catches bugs in how the wrapper plumbs `ctx`/`needs_input_grad`, not just in the kernel.

Expected output:

```
[1/3] Forward parity (atol=1e-4, rtol=1e-3)
  [OK ] p_new        max abs 1.xx e-06  max rel 1.xx e-06
  ...
[2/3] Backward parity, direct ext call (atol=1e-3, rtol=1e-2)
  [OK ] g_p          max abs 5.xx e-06  max rel 1.xx e-05
  ...
[3/3] SRBDStepFunction.apply round-trip (atol=1e-3, rtol=1e-2)
  ...
All SRBD kernel tests passed.
```

Per-tensor max absolute and max relative diff is printed for every check, so a failing line tells you which gradient drifted and by how much. Concrete numbers depend on GPU + driver; **orders of magnitude are what matter**. Drift around `1e-3` on `g_q_ref12` is the expected fast-math hit on the per-foot `sinf`/`cosf` chain — it is not a regression. Drift above the printed tolerances indicates one of:

- The extension wasn't rebuilt after a `.cu`/`.cpp` change (re-run `python setup.py build_ext --inplace`).
- A real math error was introduced in the kernel.
- You want bit-tighter parity than fast-math allows — drop `--use_fast_math` from the `nvcc` flags in `setup.py` and rebuild; the test should then pass with much smaller residuals (cost: marginally slower forward/backward).

The test requires the extension to be built and a CUDA-capable GPU. It does **not** depend on Isaac Gym and does **not** read `config.py` — the dispatch toggle is bypassed internally so the kernel itself is always exercised.

### When to enable it

- **Small `num_envs` (≤ 64)**: PyTorch is usually fine; the kernel-launch overhead of the many small ops doesn't dominate.
- **Large `num_envs` (a few hundred to a few thousand)**: the fused kernel becomes substantially faster than the PyTorch path because it replaces dozens of small dispatched ops per env with a single launch, and the per-env working set fits entirely in L2.
- **For debugging / numerical comparison**: keep `CUDA_KERNEL_SRBD = False`. The PyTorch path is the reference implementation.

### Requirements (CUDA kernel only)

- A working PyTorch CUDA install (`python -c "import torch; print(torch.cuda.is_available())"` returns `True`).
- A CUDA Toolkit on `PATH` matching your PyTorch build (`nvcc --version`).
- A C++ compiler compatible with that PyTorch build (gcc/clang on Linux, MSVC Build Tools on Windows).

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

### Q: `CUDA_KERNEL_SRBD = True` but I see the fallback warning?
A: The extension hasn't been built (or not in the current Python environment). From the project root:
```bash
python setup.py build_ext --inplace
```
This produces `srbd_cuda_ext*.so` / `*.pyd` next to `srbd.py`. After that, set `CUDA_KERNEL_SRBD = True` in `config.py` and re-run. If the build itself fails, check that `nvcc --version` works and that the CUDA Toolkit matches your PyTorch build (`python -c "import torch; print(torch.version.cuda)"`).

### Q: Do I need to rebuild the extension after every code change?
A: Only after modifying `src/srbd_cuda.cu`, `src/srbd_ext.cpp`, or `setup.py`. Changes to `srbd.py` or any other `.py` file do **not** require a rebuild — just re-run `python train.py`.

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
