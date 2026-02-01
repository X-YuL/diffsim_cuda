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

## Training

To start the training process, run:

```bash```
python many_dog_walk_vectorized.py

---

## Playback

To visualize / play the result, run:


python play_many_dog.py --num_envs 4  
python play_many_dog.py --num_envs 16  
python play_many_dog.py --num_envs 64   


