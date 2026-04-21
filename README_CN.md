# 四足机器人步态训练系统

基于 Isaac Gym 的四足机器人（Unitree Go2）步态控制训练系统，使用 SRBD（Single Rigid Body Dynamics）模型和神经网络策略进行端到端训练。

## 项目结构

```
single_dog_training/
├── config.py              # 全局配置和环境参数
├── utils_math.py          # 数学工具函数（四元数、旋转矩阵等）
├── terrain.py             # 地形创建（平面/随机起伏地形）
├── policy.py              # 神经网络策略（MLP）
├── gait.py                # 步态规划器（GaitPlanner）
├── srbd.py                # 简化刚体动力学模型（SRBDModel）
├── env.py                 # Isaac Gym 仿真环境（RealQuadEnv）
├── train.py               # 训练主循环
└── play_many_dog.py       # 策略播放脚本
```

## 功能特性

### 支持的步态
- **Stand**（站立）：四足同步
- **Trot**（小跑）：对角步态 (FL+RR, FR+RL)
- **Pace**（并行小跑）：侧对步态 (FL+RL, FR+RR)
- **Bound**（跃步）：前后腿同步
- **Gallop**（疾驰）：四足依次相位递增

### 核心技术
- **SRBD 模型**：简化的单刚体动力学，用于可微分物理仿真
- **α-对齐机制**：混合真实物理和 SRBD 预测（默认 α=0.9）
- **Raibert 落脚点规划**：基于速度反馈的自适应落脚点计算
- **多环境并行训练**：支持同时训练多个机器狗（默认 16 个）
- **GPU 加速**：使用 Isaac Gym 的 GPU 物理管线

### 损失函数
训练使用多项损失的加权组合：
- `loss_v`：速度跟踪（vx, vy）
- `loss_h`：高度保持（目标 0.35m）
- `loss_omega`：角速度正则化
- `loss_ctrl`：控制输入正则化
- `loss_gproj`：重力投影（保持机体水平）
- `loss_foot`：足端位置跟踪
- `loss_yaw`：偏航角保持

## 环境要求

### 必需依赖
- Python 3.8+
- PyTorch 1.10+
- Isaac Gym Preview 4
- NumPy
- Matplotlib
- tqdm

### 安装 Isaac Gym
```bash
# 下载 Isaac Gym Preview 4
# 解压后进入目录
cd isaacgym/python
pip install -e .
```

⚠️ **重要**：Isaac Gym 必须在 PyTorch 之前导入，代码中已处理此顺序。

## 使用方法

### 训练

```bash
# 基础训练（1000 次迭代，每次 24 步）
python3 train.py

# 训练完成后会生成：
# - quad_diffsim_srbd_align_multi_robot.pth  （模型权重）
# - quad_diffsim_srbd_align_multi_robot.pt   （TorchScript 模型，用于部署）
# - 各种训练曲线图（loss_*.png, vx_curve_*.png 等）
```

### 播放训练好的策略

```bash
# 默认播放（4 只狗，随机速度指令）
python3 play_many_dog.py

# 16 只狗一起跑
python3 play_many_dog.py --num_envs 16

# 固定速度指令（0.5 m/s）
python3 play_many_dog.py --no_rand_cmd

# 指定步态（1=trot）
python3 play_many_dog.py --gait_mode 1

# 指定权重文件
python3 play_many_dog.py --weights your_model.pth

# 运行指定步数后自动停止
python3 play_many_dog.py --max_steps 1000
```

## 配置说明

主要配置在 `config.py` 中的 `EnvCfg` 类：

### 物理参数
- `g = 9.81`：重力加速度
- `h0 = 0.35`：目标高度（米）
- `dt = 0.002`：仿真时间步长（500 Hz）

### 控制参数
- `action_hold = 5`：控制频率（100 Hz）
- `pd_kp = 60`：PD 控制器比例增益
- `pd_kd = 2`：PD 控制器微分增益

### 步态参数
- `step_freq = 1.6`：步频（Hz）
- `swing_height = 0.12`：摆动高度（米）
- `gait_mode = 1`：步态模式（-1=随机，0=stand，1=trot，2=pace，3=bound，4=gallop）

### 训练参数
- `num_envs = 16`：并行环境数
- `alpha_align = 0.9`：SRBD 对齐系数
- `train_no_aerial = True`：训练时禁用腾空段（warm-up）

### 全局开关
- `PURE_PAPER_MODE = True`：纯论文版本（无工程技巧）
- `USE_COMPLEX_TERRAIN = False`：使用随机起伏地形
- `ONLY_ITERATE_NO_RESET = True`：只在第一次迭代时 reset

## 训练输出

训练完成后会生成以下文件：

### 模型文件
- `quad_diffsim_srbd_align_multi_robot.pth`：PyTorch 模型权重
- `quad_diffsim_srbd_align_multi_robot.pt`：TorchScript 模型（用于 ROS2 部署）

### 训练曲线
- `loss_curve_srbd_align.png`：总损失曲线
- `loss_components_curve_srbd_align.png`：各项损失分量
- `vx_curve_srbd_align.png`：机体前向速度曲线
- `vx_curve_srbd_align_smooth.png`：平滑后的速度曲线
- `reward_curve_srbd_align.png`：奖励曲线

### 数据文件
- `*.npy`：各项指标的 NumPy 数组（用于后续分析）

## 代码架构

### 模块职责

- **config.py**：集中管理所有配置参数和全局开关
- **utils_math.py**：提供四元数、旋转矩阵、重力投影等数学工具
- **terrain.py**：创建平面或随机起伏地形
- **policy.py**：定义神经网络策略（36 维输入 → 256×256 → 12 维输出）
- **gait.py**：`GaitPlanner` 类，负责步态规划、相位管理、落脚点计算
- **srbd.py**：`SRBDModel` 类，实现简化刚体动力学的前向推进
- **env.py**：`RealQuadEnv` 类，封装 Isaac Gym 仿真环境
- **train.py**：训练主循环，包含损失计算、反向传播、模型保存

### 关键设计

#### GaitPlanner（步态规划器）
通过 `__getattr__` 和 `__setattr__` 代理访问环境属性，避免循环依赖：
```python
self.gait = GaitPlanner(self)
pref, stance_mask = self.gait._update_foot_targets_from_command(phases, p_foot)
```

#### SRBDModel（简化刚体动力学）
同样使用代理模式，实现可微分的物理推进：
```python
self.srbd = SRBDModel(self)
self.srbd._srbd_step(f_world=f_est, q_ref12=qref, dt=dt)
```

#### α-对齐机制
在每个训练步骤中，混合 Isaac Gym 的真实物理和 SRBD 预测：
```python
env.srbd_p = env.base_pos + alpha * (env.srbd_p - env.srbd_p.detach())
env.srbd_v = env.base_lin + alpha * (env.srbd_v - env.srbd_v.detach())
```

## 调试功能

### 开局前翻定位打印
在 `config.py` 中设置：
```python
DBG_INIT_FALL = True
DBG_INIT_FALL_STEPS = 250  # 只打印前 250 步
DBG_INIT_FALL_EVERY = 5    # 每 5 步打印一次
DBG_INIT_FALL_ENV = 0      # 只看第 0 只狗
```

### 梯度流分析
训练时会自动打印每层的梯度范数，用于诊断梯度消失/爆炸问题。

### 方向检查
每 20 次迭代打印一次机体速度方向，验证速度跟踪是否正确。

## 常见问题

### Q: 训练时机器狗一直前翻？
A: 检查以下几点：
1. `train_no_aerial = True`（禁用腾空段）
2. `alpha_align = 0.9`（足够大的对齐系数）
3. 步频不要太高（建议 1.5-1.8 Hz）
4. 检查 `last_contact_z` 是否正确更新

### Q: Isaac Gym 导入失败？
A: 确保 Isaac Gym 在 PyTorch 之前导入。代码中已处理，但如果仍有问题，检查：
```python
# 正确顺序
from isaacgym import gymapi
import torch

# 错误顺序
import torch
from isaacgym import gymapi  # 会报错
```

### Q: 训练速度慢？
A:
1. 确保 `use_gpu_pipeline = True`
2. 增加 `num_envs`（更多并行环境）
3. 关闭 `use_viewer`（训练时不渲染）
4. 使用更快的 GPU

### Q: 如何部署到真实机器人？
A: 训练完成后使用 TorchScript 模型：
```python
model = torch.jit.load("quad_diffsim_srbd_align_multi_robot.pt")
action = model(observation)  # (1, 36) -> (1, 12)
```

## 引用

如果使用本代码，请引用相关论文（根据你的实际论文填写）：

```bibtex
@article{your_paper,
  title={Your Paper Title},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## 许可证

（根据你的项目许可证填写）

## 联系方式

（根据你的联系方式填写）

---

**注意**：本项目基于 Isaac Gym Preview 4，仅用于研究和教育目的。
