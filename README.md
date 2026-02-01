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

---

## 注意事项：

61行 那里是训练时是否使用复杂地形的开关  
<img width="504" height="88" alt="image" src="https://github.com/user-attachments/assets/a9eba6c3-75a5-49dd-a4b5-8fdb6d65caa8" />


297 - 298 行设置频率范围, step_freq_from_cmd 是否频率随着速度变化而变化  
<img width="1059" height="145" alt="image" src="https://github.com/user-attachments/assets/5bfb0dc0-7762-47de-a400-e0051808da56" />


329 行 决定训练使用哪种步态  
<img width="697" height="257" alt="image" src="https://github.com/user-attachments/assets/028116eb-41a3-4ce0-8f90-2bf9f46387f7" />


2590 - 2591 行 决定训练的随机速度指令范围  
<img width="567" height="37" alt="image" src="https://github.com/user-attachments/assets/d5c8d9b0-2138-44b0-8db6-2bfae29bb383" />









