# RonGenZ

Minimal scripts for vectorized training and playback.

---

## Environment Setup

### Python Environment
The code is tested with the following environment:

- **PyTorch**: 2.2.2+cu118
- **Python**: 3.8.20
- **CUDA**: 11.8
- **Isaac Gym**: 1.0rc4

It should be compatible with similar PyTorch/CUDA versions.

---

## Files

- `many_dog_walk_vectorized.py`  
  Vectorized training / rollout script.

- `play_many_dog.py`  
  Playback / visualization script.

---

## Installation

> If your project has dependencies, list them here.  
> For example, create a venv/conda env and install requirements.

```bash
# (optional) create env
python -m venv .venv
source .venv/bin/activate

# (optional) install deps
pip install -r requirements.txt

