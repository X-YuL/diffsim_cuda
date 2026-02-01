# -*- coding: utf-8 -*-
"""
DiffSim Quadruped Toy (multi-robot) — Isaac Gym + SRBD 可微代理 + α 对齐
- 支持 B 只机器狗并行：IsaacGym 多 env + 每只狗一套 SRBD 状态
- 代码整体结构与单狗版保持一致，只是所有状态 / 动作都加了 batch 维 (B, ·)

A–G 改动说明：
A) 站立掩码平滑：phase + 接触混合，并强制“至少 2 足承载”兜底
B) 力估计岭回归增强：SVD + 奇异值地板 + 切向力 ≤ μFz + Fz 下限/上限
C) 改进 Raibert 落脚点：
   p_land = p_hip + 0.5 * T_stance * v_des + k * (v_des - v_now) + x_bias
   ★ 此处 v_des 改为“机体前向 vx_star 投到世界系”，反馈用“机体系 vx 误差”沿机体前向修正
D) 真实–代理状态对齐：数值来自 Isaac，稳定性来自 SRBD（对齐 v 和 p_z）
E) 动作幅度与关节变化率限制：
   delta_q = tanh(delta_q) * 0.30，pos_target 每步限制 0.04

（新增调参）
F) reset 时可选给 base 一个机体前向初速度（工程版），PURE_PAPER_MODE 下不加
G) 加强目标速度项权重 a1（更强调“机体前向 vx_body=0.2 m/s”）

（新）Multi-dog 改动：
- EnvCfg 新增 num_envs：同时跑 B 只狗
- IsaacGym: 创建 num_envs 个 env，每个 env 一只 go2
- 所有状态/动作/观测统一改为 batch 形式：(B, ·)
- 每只狗一套 SRBD 状态 srbd_p/v/q/w，估计足端力也是 (B, 4, 3)
"""

# ===============================
# Swing cycloid (教材 9.3 / 式 9.14-9.15) 最小补丁：
# - 记录每条腿“离地瞬间”足端起点 p0（world）
# - 摆动期用摆线解析式从 p0 -> touchdown p1
# - 支撑期仍锁死 last_contact（你原本的逻辑保留）
# ===============================

import os, sys, math, random
from dataclasses import dataclass
from typing import Optional
import numpy as np

# ===============================
# ⭐ 论文纯净版开关
# ===============================
PURE_PAPER_MODE = True
# True  = 最纯净论文版（无工程 trick）
# False = 工程版（带初速度、动作平滑等）

# ===============================
# ⭐ 开局前翻定位打印开关（只打印 env0 前若干步）
# ===============================
DBG_INIT_FALL = True
DBG_INIT_FALL_STEPS = 250     # 只打印前 N 个 env.step()（每次 reset 后 t 从 0 开始）
DBG_INIT_FALL_EVERY = 5       # 每隔多少步打印一次
DBG_INIT_FALL_ENV = 0         # 只看哪一只狗（env index）


# ⭐ 是否只在 iter=0 reset
ONLY_ITERATE_NO_RESET = True
# True 不reset
# False reset

# ⭐ 是否使用复杂随机地形
USE_COMPLEX_TERRAIN = False
# True  = 使用随机起伏 heightfield 地形（近似无限平面）
# False = 使用原来的平面 ground plane

# ---------------- Isaac Gym ----------------
try:
    from isaacgym import gymapi, gymtorch,terrain_utils
    ISAAC_AVAILABLE = True
    print("[Info] Isaac Gym import OK")
except Exception as e:
    ISAAC_AVAILABLE = False
    print("[Warning] Isaac Gym import failed:", repr(e))
    print("   PYTHONPATH =", os.environ.get("PYTHONPATH"))
    print("   LD_LIBRARY_PATH =", os.environ.get("LD_LIBRARY_PATH"))
    print("   sys.path head =", sys.path[:3])

# ---------------- Torch ----------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- Utils ----------------
def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def moving_average(x: np.ndarray, k: int = 25) -> np.ndarray:
    if k <= 1: return x
    w = np.ones(k) / k
    return np.convolve(x, w, mode="same")

# 四元数 wxyz -> 旋转矩阵（非 batch 版）
def quat_to_rot(qw, qx, qy, qz, device):
    r00 = 1 - 2 * (qy*qy + qz*qz)
    r01 = 2 * (qx*qy - qz*qw)
    r02 = 2 * (qx*qz + qy*qw)
    r10 = 2 * (qx*qy + qz*qw)
    r11 = 1 - 2 * (qx*qx + qz*qz)
    r12 = 2 * (qy*qz - qx*qw)
    r20 = 2 * (qx*qz - qy*qw)
    r21 = 2 * (qy*qz + qx*qw)
    r22 = 1 - 2 * (qx*qx + qy*qy)
    return torch.stack([
        torch.stack([r00, r01, r02]),
        torch.stack([r10, r11, r12]),
        torch.stack([r20, r21, r22]),
    ]).to(device)

# 将重力加速度投影到机体系 -> 用于观测和 loss 中的重力惩罚
def project_gravity_to_body(q_wxyz, g, device):
    # q_wxyz = (w,x,y,z)
    qw, qx, qy, qz = q_wxyz
    R = quat_to_rot(qw, qx, qy, qz, device)
    g_w = torch.tensor([0.0, 0.0, -g], dtype=torch.float32, device=device)
    return R.t().matmul(g_w)  # (3,)

# 欧拉角 -> 四元数
def quat_from_rpy(roll: float, pitch: float, yaw: float) -> "gymapi.Quat":
    # gymapi.Quat(x, y, z, w)
    cr, sr = math.cos(roll*0.5),  math.sin(roll*0.5)
    cp, sp = math.cos(pitch*0.5), math.sin(pitch*0.5)
    cy, sy = math.cos(yaw*0.5),   math.sin(yaw*0.5)
    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    return gymapi.Quat(x, y, z, w)

# 世界系 -> 机体系旋转（单个 q、v）
def quat_rotate_inverse_wxyz(q_wxyz, v, device):
    """
    把向量 v 从世界系旋到机体系：v_body = R(q)^T v_world
    q_wxyz: (4,), v: (3,)
    """
    qw, qx, qy, qz = q_wxyz
    R = quat_to_rot(qw, qx, qy, qz, device)  # world <- body
    return (R.t() @ v.view(3,)).view(3,)

# PhysX 稳定性设置
def _setup_physx_stable(sim_params, use_gpu=True):
    if hasattr(sim_params, "substeps"):
        sim_params.substeps = 3
    if not hasattr(sim_params, "physx"):
        return
    ph = sim_params.physx
    if hasattr(ph, "num_position_iterations"): ph.num_position_iterations = 12
    if hasattr(ph, "num_velocity_iterations"): ph.num_velocity_iterations = 2
    if hasattr(ph, "solver_type"):
        if hasattr(gymapi, "SOLVER_TGS"):
            ph.solver_type = gymapi.SOLVER_TGS
        else:
            try: ph.solver_type = 1
            except Exception: pass
    if hasattr(ph, "use_gpu"): ph.use_gpu = bool(use_gpu)
    if hasattr(ph, "rest_offset"): ph.rest_offset = 0.0
    if hasattr(ph, "contact_offset"): ph.contact_offset = 0.01
    if hasattr(ph, "bounce_threshold_velocity"): ph.bounce_threshold_velocity = 0.2
    if hasattr(ph, "max_depenetration_velocity"): ph.max_depenetration_velocity = 1.0
    if hasattr(ph, "default_buffer_size_multiplier"): ph.default_buffer_size_multiplier = 2.0
    if hasattr(ph, "enable_stabilization"): ph.enable_stabilization = True
    if hasattr(ph, "enable_ccd"): ph.enable_ccd = True

# ================== 地形创建工具 ==================
def create_ground_plane(gym, sim):
    """原来的平面地面封装成一个小函数。"""
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    plane_params.static_friction  = 1.0
    plane_params.dynamic_friction = 1.0
    plane_params.restitution      = 0.0
    gym.add_ground(sim, plane_params)

def create_random_rough_terrain(gym, sim):
    """
    使用 isaacgym.terrain_utils 创建一大块随机起伏地形，
    再转换为三角网格加到 PhysX 里。

    为了“近似无限”，这里做的是 80m x 80m 的大地形，
    中心对齐 (0,0)，狗一开始出生在中间附近。
    """
    # 尺度参数
    horizontal_scale = 0.25   # 每个 heightfield 格子 0.25m
    vertical_scale   = 0.005  # 每个高度单位 0.005m

    terrain_size = 200       # 80m x 80m
    num_rows = int(terrain_size / horizontal_scale)
    num_cols = int(terrain_size / horizontal_scale)

    # 创建一块子地形：全是随机起伏
    sub = terrain_utils.SubTerrain(
        terrain_name="random_uniform",
        width=num_rows,
        length=num_cols,
        vertical_scale=vertical_scale,
        horizontal_scale=horizontal_scale
    )

    # 随机高度范围（单位：米）
    # 范围不要太大，避免一出生就被埋脚 / 太高跳不下来
    #min_h = -0.15
    #max_h =  0.15
    min_h = -0.02
    max_h =  0.02

    terrain_utils.random_uniform_terrain(
        sub,
        min_height=min_h,
        max_height=max_h,
        step=0.02,           # 台阶高度粒度 ~3cm
        #downsampled_scale=0.5  # 降采样尺度，越大越平缓
        downsampled_scale=0.25
    )

    heightfield = sub.height_field_raw   # (num_rows, num_cols) int16

    # 高度场 -> 三角网格
    vertices, triangles = terrain_utils.convert_heightfield_to_trimesh(
        heightfield,
        horizontal_scale=horizontal_scale,
        vertical_scale=vertical_scale,
        slope_threshold=1.5,
    )

    tm_params = gymapi.TriangleMeshParams()
    tm_params.nb_vertices  = vertices.shape[0]
    tm_params.nb_triangles = triangles.shape[0]

    # 把地形中心对齐到 (0,0)
    tm_params.transform.p.x = -terrain_size * 0.5
    tm_params.transform.p.y = -terrain_size * 0.5
    #tm_params.transform.p.z = 0.0
    tm_params.transform.p.z = -0.03

    gym.add_triangle_mesh(
        sim,
        vertices.flatten(),
        triangles.flatten(),
        tm_params
    )

    print(f"[Terrain] Random rough terrain created: size = {terrain_size}m x {terrain_size}m")

# ---------------- Policy ----------------
class Policy(nn.Module):
    """神经网络策略 : 输入36维观测 -> 256 * 256 -> 输出12维关节角 offset"""
    def __init__(self, dim_obs=36, dim_action=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_obs, 256), nn.LeakyReLU(0.05),
            nn.Linear(256, 256),     nn.LeakyReLU(0.05),
            nn.Linear(256, dim_action)
        )
        with torch.no_grad():
            nn.init.normal_(self.net[-1].weight, std=1e-2)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, s, h=None):
        return self.net(s), None

    def reset(self): pass

# ---------------- Config ----------------
@dataclass
class EnvCfg:
    # physics
    g: float = 9.81
    h0: float = 0.35   # 0.30
    dt: float = 0.002  # 500 Hz

    # control
    use_gpu_pipeline: bool = True
    use_viewer: bool = False  #训练时是否渲染？True - 渲染但会满慢很多 ； False - 不渲染
    action_hold: int = 5      # 100 Hz 控制

    pd_kp: float = 60       #60
    pd_kd: float = 2        #2
    q_default: tuple = (0.0, 0.9, -1.20)
    #q_default: tuple = (0.0, 0.8, -1.50)
 
    # gait
    use_paper_raibert: bool = True # True=论文原始式子, False=工程增强版（机体 vx tracking）
    step_freq: float = 2.2
    
    # 随机步频开关 - True 开； False 关（固定 step_freq）
    rand_step_freq: bool = False       # True: sample step_freq_B on reset/reset_envs; False: use constant step_freq
    step_freq_min: float = 1.4        # 1.4
    step_freq_max: float = 3.2         #3.2
    
    # 步频随速度u开关 - True 开； False 关（固定 step_freq_B = 2.2）
    step_freq_from_cmd: bool = True  # True: override step_freq_B each step from |vx_star|
    
    # Raibert 参数
    #swing_height: float =  0.012        #固定抬腿高度 ： 0.025
    #k_raibert: float = 0.55
    #x_bias: float = 0
    # 训练前期建议：抬脚高一点 + touchdown 反馈小一点（否则很容易把落脚点推到腿够不到的位置）.03          # 额外前移落脚偏置（m）
    swing_height: float = 0.025

    

    k_raibert: float = 0.12
    raibert_fb_clip: float = 0.15       # m，touchdown 反馈项每轴限幅
    x_bias: float = 0.0                # 额外前移落脚偏置（m）

    # 训练 warm-up：先禁用腾空段，等学稳了再打开（bound/gallop/running-trot）
    train_no_aerial: bool = True
    
    # ✅ (新增) 用于 _update_gait_from_cmd 的 swing height 上限；必须 >= swing_height
    swing_height_max: float = 0.05
 
    # gait selection 开关：
    # -1: 每个 env 随机一个步态（stand / trot / pace / bound / gallop）
    #  0: stand
    #  1: trot
    #  2: pace
    #  3: bound
    #  4: gallop
    gait_mode: int = 1
    trot_style: str = "normal"   # "normal" | "walk" | "run"
    #trot_style = "normal"  # β=0.5 (图9.2a)
    #trot_style = "walk"  # β=0.6 (图9.2b)
    #trot_style = "run"   # β=0.4 (图9.2c)

    z_time_constant: float = 0.08   # stance 时脚高度收敛时间常数，越小脚越快贴到目标高度

    # settle steps
    settle_steps_init: int = 60
    settle_steps_reset: int = 40

    # SRBD params
    
    m: float = 12.0
    Ixx: float = 0.25
    Iyy: float = 0.90
    Izz: float = 1.00



    alpha_align: float = 0.9
    use_strict_alpha_align: bool = True

    # termination
    term_penalty: float = 200.0

    #============================================
    # 随机速度指令 开关 - True 开； False 关（固定 0.2 m/s）
    rand_cmd: bool = True       # True: sample cmd_B on reset/reset_envs; False: use constant cmd
    # trot/pace    0.5 - 1 m/s
    # bound/gallop 1 - 2 m/s
    vx_min: float = +0.5   #0.5
    vx_max: float = +1.0   #1.0
    # 侧向速度、偏航角速度命令范围（默认全 0，你以后想让狗学横移 / 转弯再改）
    vy_min: float = 0
    vy_max: float = 0
    yaw_min: float = 0        # [rad/s]
    yaw_max: float = 0
    #============================================


    # contact & friction
    contact_thresh_n: float = 8.0
    mu_tangent: float = 0.6
    fz_min: float = 20.0
    fz_max: float = 250.0

    # 并行环境数 = 并行机器狗数
    num_envs: int = 16

    cmd_deadzone: float = 0.05   # m/s，认为“停”的阈值

    contact_on_n: float  = 20.0
    contact_off_n: float = 10.0

# ========= Real quadruped Isaac Gym env (multi robot) =========
class RealQuadEnv:
    def __init__(self, cfg: EnvCfg, device="cuda" if torch.cuda.is_available() else "cpu"):
        assert ISAAC_AVAILABLE, "需要 Isaac Gym 运行此真实四足环境。"

        self.cfg = cfg
        self.device = torch.device(device)
        self.B = int(cfg.num_envs)  # 并行机器狗数量

        self._render_cnt = 0
        self.render_every = 1   # 10~50 都可以试试

        # ★ 高层速度命令：cmd_rand = [vx_cmd, vy_cmd, yaw_rate_cmd]
        self.cmd_rand = torch.zeros(self.B, 3, device=self.device)   # (B,3)
        self.vx_star  = torch.zeros(self.B, device=self.device)      # 仅保留一个别名给 Raibert / loss 用


        # thight - 大腿 ； calf - 小腿
        # 12 个受控关节的 key 名称顺序
        self.key12 = [
            "FL_hip","FL_thigh","FL_calf",
            "FR_hip","FR_thigh","FR_calf",
            "RL_hip","RL_thigh","RL_calf",
            "RR_hip","RR_thigh","RR_calf",
        ]

        # 默认站立姿态

        """
        #目前 trot 适应 1 m/s 的姿态
        self.q_default_np = np.array([
            +0.1, +0.8, -1.50,
            -0.1, +0.8, -1.50,
            +0.1, +0.8, -1.50,
            -0.1, +0.8, -1.50
        ], dtype=np.float32)
        """
       
        """
        #目前最好 - 全姿态通用！！！ 
        #在这个姿态下，trot/bound/gallop 都能跑 1m/s ; pace 能跑 0.7m/s
        #固定频率 4Hz 下测试；目标速度 1m/s
        self.q_default_np = np.array([
            +0.0, +1.0, -1.50,
            -0.0, +1.0, -1.50,
            +0.0, +1.0, -1.50,
            -0.0, +1.0, -1.50
        ], dtype=np.float32)
        """
        """
        self.q_default_np = np.array([
            +0.0, +0.77, -1.20,
            -0.0, +0.77, -1.20,
            +0.0, +0.77, -1.20,
            -0.0, +0.77, -1.20
        ], dtype=np.float32)
        """

        self.q_default_np = np.array([
             0.0, 0.86, -1.40,
            -0.0, 0.86, -1.40,
             0.0, 0.86, -1.40,
            -0.0, 0.86, -1.40,
        ], dtype=np.float32)
        
        
        

        # 相位：FL、FR、RL、RR -> 四足交替步态
        #self.leg_phase_offsets = torch.tensor(
        #    [0.0, math.pi, math.pi, 0.0], dtype=torch.float32, device=self.device
        #)
        
        # === 多步态：定义不同步态的相位模式（顺序：FL, FR, RL, RR） ===
        # 0: stand   - 四条腿几乎同步
        # 1: trot    - 对角小跑 (FL+RR, FR+RL)
        # 2: pace    - 并行小跑 / 侧对小跑 (FL+RL, FR+RR)
        # 3: bound   - 跃步 (前腿同步, 后腿同步)
        # 4: gallop  - 疾驰 (FL, FR, RL, RR 相位依次递增)
        stand = torch.zeros(4, dtype=torch.float32, device=self.device)

        #trot  = torch.tensor([0.0, math.pi, math.pi, 0.0], dtype=torch.float32, device=self.device)
        trot  = torch.tensor([0.0, math.pi, math.pi, 0.0], dtype=torch.float32, device=self.device)

        pace  = torch.tensor([0.0, math.pi, 0.0, math.pi], dtype=torch.float32, device=self.device)
        
        #bound = torch.tensor([0.0, 0.0, 0.5*math.pi, 0.5*math.pi], dtype=torch.float32, device=self.device)
        bound = torch.tensor([0.0, 0.0, math.pi, math.pi], dtype=torch.float32, device=self.device)
        
        # gallop: 前腿同步，后腿同步，但相位比前腿滞后 90°
        #         这样会出现「后腿起跳→四腿离地→前腿落地」的序列，区别于 bound 的 180° 偏移
        gallop = torch.tensor(
            [ 0, 0.1*2* math.pi,  math.pi, 1.1 * math.pi],  # FL, FR, RL, RR
            dtype=torch.float32,
            device=self.device
        )
    
        self.gait_table = torch.stack([stand, trot, pace, bound, gallop], dim=0)   # (G,4)
        self.num_gaits = self.gait_table.shape[0]

        # 兼容旧代码：默认 trot
        self.leg_phase_offsets = trot.clone()                              # (4,)

        # 从 cfg 里读 gait_mode
        self.gait_mode = getattr(cfg, "gait_mode", -1)

        # 初始化每个 env 的 gait_id
        if self.gait_mode < 0:
            # -1: 每个 env 随机，真正的采样放在 reset() 里做
            self.gait_ids = torch.ones(self.B, dtype=torch.long, device=self.device)
        else:
            # 固定所有 env 同一个 gait
            gid = int(self.gait_mode)
            gid = max(0, min(self.num_gaits - 1, gid))    # clamp 到 [0, num_gaits-1]
            self.gait_ids = torch.full(
                (self.B,), gid, dtype=torch.long, device=self.device
            )

        # (B,4) 每个 env 当前的四条腿相位偏移
        self.leg_phase_offsets_B = self.gait_table[self.gait_ids]


        # === Isaac Gym 实例 & 物理参数 ===
        self.gym = gymapi.acquire_gym()
        sim_params = gymapi.SimParams()
        sim_params.dt = self.cfg.dt
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -self.cfg.g)
        sim_params.use_gpu_pipeline = self.cfg.use_gpu_pipeline
        _setup_physx_stable(sim_params, use_gpu=True)

        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
        assert self.sim is not None, "create_sim 失败"

        # ===== 地形 / 地面：根据开关选择 =====
        if USE_COMPLEX_TERRAIN:
            print("[Terrain] USE_COMPLEX_TERRAIN=True -> 使用随机起伏 heightfield 地形")
            create_random_rough_terrain(self.gym, self.sim)
        else:
            print("[Terrain] USE_COMPLEX_TERRAIN=False -> 使用平面 ground plane")
            create_ground_plane(self.gym, self.sim)


        # 创建多 env，每个 env 一只 go2
        spacing = 2.0
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        num_per_row = int(math.ceil(math.sqrt(self.B)))

        self.envs = []
        self.actor_handles = []
        self.actor_indices = []
        # env origins（SIM 世界坐标系下，每个 env 的原点）
        self.env_origins = torch.zeros(self.B, 3, device=self.device, dtype=torch.float32)



        # 加载资产
        ASSET_ROOT = "/home/rongenz/unitree_rl_gym-main/resources/robots/go2/urdf"
        ASSET_FILE = "go2.urdf"

        asset_opts = gymapi.AssetOptions()
        asset_opts.fix_base_link = False
        asset_opts.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)
        asset_opts.thickness = 0.01
        asset_opts.angular_damping = 0.01
        asset_opts.armature = 0.03
        for field in ("collapse_fixed_joints", "use_mesh_materials"):
            if hasattr(asset_opts, field):
                setattr(asset_opts, field, True if field=="collapse_fixed_joints" else False)
        for field in ("enable_self_collisions", "self_collisions", "use_self_collisions"):
            if hasattr(asset_opts, field):
                setattr(asset_opts, field, True)
                break
        if hasattr(asset_opts, "flip_visual_attachments"):
            asset_opts.flip_visual_attachments = True

        self.robot_asset = self.gym.load_asset(self.sim, ASSET_ROOT, ASSET_FILE, asset_opts)
        assert self.robot_asset is not None, f"加载 {ASSET_FILE} 失败，请检查路径。"

        # 资产关节信息
        self.dof_count = self.gym.get_asset_dof_count(self.robot_asset)
        raw_names = []
        for i in range(self.dof_count):
            n = self.gym.get_asset_dof_name(self.robot_asset, i)
            n = n.decode("utf-8") if isinstance(n, bytes) else n
            raw_names.append(n)

        # 资产 -> key12 映射
        def to_key(name: str):
            s = name.lower()
            if   s.startswith("fl_"): leg = "FL"
            elif s.startswith("fr_"): leg = "FR"
            elif s.startswith("rl_"): leg = "RL"
            elif s.startswith("rr_"): leg = "RR"
            else: return None
            if   "_hip_"   in s: joint = "hip"
            elif "_thigh_" in s: joint = "thigh"
            elif "_calf_"  in s: joint = "calf"
            else: return None
            return f"{leg}_{joint}"

        name2idx = {}
        for i, n in enumerate(raw_names):
            k = to_key(n)
            if k and (k not in name2idx): name2idx[k] = i

        print("\n[DBG] Asset DOFs in order:")
        for i, n in enumerate(raw_names):
            print(f"  {i:02d}: {n}")

        print("\n[DBG] Mapping to key12:")
        for k in self.key12:
            print(f"  {k:12s} -> asset dof idx {name2idx.get(k)}")

        ctrl_idx_list = [name2idx[k] for k in self.key12]
        self.ctrl_idx = np.array(ctrl_idx_list, dtype=np.int32)
        self.ctrl_idx_t = torch.as_tensor(self.ctrl_idx, device=self.device, dtype=torch.long)
        self.ctrl_sign = torch.ones(12, device=self.device, dtype=torch.float32)

        print("[DBG] ctrl_idx (key12 -> raw_names 索引):", self.ctrl_idx.tolist())
        self.ctrl_leg_tags = [k.split("_")[0].upper() for k in self.key12]

        def _leg_of_key(k: str) -> str:
            return k.split("_")[0].upper()

        legs_seq = [_leg_of_key(k) for k in self.key12]
        expected = ["FL"]*3 + ["FR"]*3 + ["RL"]*3 + ["RR"]*3
        if legs_seq != expected:
            print(f"[WARN] key12 的顺序不是 [FL*3, FR*3, RL*3, RR*3]，当前:", legs_seq)
            for leg in ("FL","FR","RL","RR"):
                leg3idxs = [i for i, k in enumerate(self.key12) if _leg_of_key(k) == leg]
                print(f"[HINT] {leg} 在 key12 中的下标:", leg3idxs)
        else:
            print(f"[OK] key12 顺序为 [FL*3, FR*3, RL*3, RR*3]")

        # 解析刚体名称，建立 feet_local
        rb_count = self.gym.get_asset_rigid_body_count(self.robot_asset)
        self.rb_count = rb_count
        rb_names = []
        for j in range(rb_count):
            nm = self.gym.get_asset_rigid_body_name(self.robot_asset, j)
            nm = nm.decode("utf-8") if isinstance(nm, bytes) else nm
            rb_names.append((j, nm))
        rb_names_lower = [(j, nm.lower()) for (j, nm) in rb_names]

        def _pick_leg_body(leg_prefix: str):
            keys = ("foot", "toe", "sole", "ankle")
            cands = [(j, nm) for (j, nm) in rb_names_lower
                     if nm.startswith(leg_prefix + "_") and any(k in nm for k in keys)]
            for prefer in ("foot", "toe", "sole", "ankle"):
                for (j, nm) in cands:
                    if prefer in nm:
                        return j
            return cands[0][0] if cands else None

        leg_order = ["fl", "fr", "rl", "rr"]
        leg2body = {leg: _pick_leg_body(leg) for leg in leg_order}
        missing_feet = [leg for leg, idx in leg2body.items() if idx is None]
        if missing_feet:
            print("[ERR] 找不到这些腿对应的足部刚体:", missing_feet)
            print("[HINT] 当前刚体名称:", [nm for _, nm in rb_names])
            raise AssertionError("feet_local 解析失败，请完善刚体命名匹配规则。")

        self.feet_local = [leg2body["fl"], leg2body["fr"], leg2body["rl"], leg2body["rr"]]
        print("[INFO] feet_local rigid bodies (FL,FR,RL,RR idx):", self.feet_local)

        # 创建 env + actor
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0, 0, self.cfg.h0)
        yaw0 = 0.0
        pose.r = quat_from_rpy(0.0, 0.0, yaw0)

        actor_name = "go2"
        for env_id in range(self.B):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.envs.append(env_ptr)
            
            o = self.gym.get_env_origin(env_ptr)  # gymapi.Vec3
            self.env_origins[env_id] = torch.tensor([o.x, o.y, o.z], device=self.device)

            actor_handle = self.gym.create_actor(env_ptr, self.robot_asset, pose, actor_name, env_id, 1)
            self.actor_handles.append(actor_handle)
            actor_index = self.gym.get_actor_index(env_ptr, actor_handle, gymapi.DOMAIN_SIM)
            self.actor_indices.append(actor_index)

        #self.actor_indices_t = torch.as_tensor(self.actor_indices, device=self.device, dtype=torch.long)
        self.actor_indices_t = torch.as_tensor(self.actor_indices, device=self.device, dtype=torch.int32)

        # DOF PD 属性（对所有 env 的 actor 相同）
        props = self.gym.get_actor_dof_properties(self.envs[0], self.actor_handles[0])
        props["driveMode"][:] = gymapi.DOF_MODE_POS
        kp, kd = self.cfg.pd_kp, self.cfg.pd_kd
        props["stiffness"][:] = 0.0
        props["damping"][:]   = 0.0
        props["stiffness"][self.ctrl_idx] = kp
        props["damping"][self.ctrl_idx]   = kd

        for env_ptr, actor_handle in zip(self.envs, self.actor_handles):
            self.gym.set_actor_dof_properties(env_ptr, actor_handle, props)

        chk = self.gym.get_actor_dof_properties(self.envs[0], self.actor_handles[0])
        print("driveMode unique (after set):", set(chk["driveMode"].tolist()))
        print("stiffness range:", float(chk["stiffness"].min()), "–", float(chk["stiffness"].max()))
        print("damping   range:", float(chk["damping"].min()), "–", float(chk["damping"].max()))
        lo12 = chk["lower"][self.ctrl_idx]; hi12 = chk["upper"][self.ctrl_idx]
        eff12 = chk["effort"][self.ctrl_idx] if "effort" in chk.dtype.names else None
        print("[DBG] lower[12] =", np.round(lo12, 6))
        print("[DBG] upper[12] =", np.round(hi12, 6))
        if eff12 is not None: print("[DBG] effort[12] =", np.round(eff12, 3))

        self.q_default_full = np.zeros(self.dof_count, dtype=np.float32)
        self.q_default_full[self.ctrl_idx] = self.q_default_np

        self.gym.prepare_sim(self.sim)

        # 全局 DOF 数
        self.sim_dof_count = self.gym.get_sim_dof_count(self.sim)
        assert self.sim_dof_count == self.B * self.dof_count, \
            f"sim_dof_count={self.sim_dof_count}, B*dof_count={self.B*self.dof_count}"

        # DOF 目标张量：一维 [B*dof_count]，再 view 成 (B,dof_count) 使用
        self.pos_targets = torch.zeros(self.sim_dof_count, dtype=torch.float32, device=self.device).contiguous()
        self.pos_targets_batch = self.pos_targets.view(self.B, self.dof_count)
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.pos_targets))

        # 状态张量包装（带 batch 视图）
        _dof_state = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state_t = gymtorch.wrap_tensor(_dof_state)
        self.dof_state_view = self.dof_state_t.view(self.B, self.dof_count, 2)

        _root = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_state = gymtorch.wrap_tensor(_root).view(self.B, 13)

        _rb = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_state_t = gymtorch.wrap_tensor(_rb).view(self.B, self.rb_count, 13)

        _jac = self.gym.acquire_jacobian_tensor(self.sim, actor_name)
        assert _jac is not None, "acquire_jacobian_tensor('go2') failed"
        self.jacobian = gymtorch.wrap_tensor(_jac)  # 将在 foot_jacobians 里 reshape

        _cf = self.gym.acquire_net_contact_force_tensor(self.sim)
        assert _cf is not None, "acquire_net_contact_force_tensor() failed"
        self.net_cf = gymtorch.wrap_tensor(_cf).view(self.B, self.rb_count, 3)

        # 关节限位（资产级）
        props_now = chk
        self._lo_slice = torch.as_tensor(props_now["lower"], device=self.device, dtype=torch.float32)
        self._hi_slice = torch.as_tensor(props_now["upper"], device=self.device, dtype=torch.float32)

        # local_targets: (B, dof_count)
        base_local = torch.as_tensor(self.q_default_full, device=self.device, dtype=torch.float32)
        self.local_targets = base_local.view(1, -1).repeat(self.B, 1)
        self.pos_targets_batch[:] = self.local_targets
        self._commit_pos_targets()

        # 先跑一小段时间稳定
        for _ in range(self.cfg.settle_steps_init):
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

        self._sanity_check_io()
        self._last_action = torch.zeros(self.B, 12, device=self.device)

        self.last_reset_yaw = torch.zeros(self.B, device=self.device)

        # Viewer 只看第 0 只狗
        self.viewer = None
        if self.cfg.use_viewer:
            cam_props = gymapi.CameraProperties()
            self.viewer = self.gym.create_viewer(self.sim, cam_props)
            self.gym.viewer_camera_look_at(self.viewer, None,
                                           gymapi.Vec3(2.0, 2.0, 1.2),
                                           gymapi.Vec3(0.0, 0.0, 0.3))

        self.cam_follow = False
        self.cam_dist   = 2.5
        self.cam_height = 1.0
        self.cam_smooth = 0.15
        self._cam_eye = None
        self._cam_tgt = None

        # 初始化缓存 + SRBD
        self._update_cache()
        self._srbd_init_from_isaac()
        #self.last_contact_z = torch.zeros(self.B, 4, device=self.device)  # 存每条腿最近接触高度
        # ===== last-contact cache (world frame) =====
        # 存每条腿最近一次“触地时刻”的足端位置（世界系），用于：stance 足端锁死（符合教材/图9.2描述）
        self.last_contact_z  = torch.zeros(self.B, 4, device=self.device)     # (B,4)
        self.last_contact_xy = torch.zeros(self.B, 4, 2, device=self.device)  # (B,4,2)
        
        # ===== liftoff cache (world frame) =====
        # 存每条腿最近一次“离地瞬间”的足端位置（世界系），用于：摆线轨迹起点 (x0,y0,z0)
        self.last_liftoff_xyz = torch.zeros(self.B, 4, 3, device=self.device) # (B,4,3)
        # 上一时刻 stance_mask（用于检测 stance->swing 的离地边沿）
        self.prev_stance_mask = torch.ones(self.B, 4, 1, device=self.device)  # (B,4,1)

        # ===== contact-edge cache (world frame) =====
        # 用于检测 touchdown（prev_contact=0 -> contact=1），避免 swing 期“擦地”污染 last_contact_*
        self.prev_contact_flags = torch.zeros(self.B, 4, 1, device=self.device)  # (B,4,1)



        # 初始化：用当前足端位置填充 last_contact_*
        with torch.no_grad():
            p_foot0 = self.foot_positions()           # (B,4,3)
            self.last_contact_xy[:] = p_foot0[..., 0:2]
            self.last_contact_z[:]  = p_foot0[..., 2]
            self.last_liftoff_xyz[:] = p_foot0
            self.prev_contact_flags[:] = self.contact_flags()

        # ===== stride debug (env0) =====
        self._stride_last_td_xy0 = torch.zeros(4, 2, device=self.device)          # (4,2)
        self._stride_have_td0    = torch.zeros(4, dtype=torch.bool, device=self.device)
        self._stride_count0      = torch.zeros(4, dtype=torch.long, device=self.device)

        # 可选：只打印前 N 次 touchdown（防刷屏）；设 None 表示不限制
        self._stride_print_limit = 200

        self._stride_last_td_step0 = torch.full((4,), -10_000, device=self.device, dtype=torch.long)



    def _sanity_check_io(self):
        # 只是做 shape 检查和 debug 打印
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        J = self.foot_jacobians()          # (B, 4, 3, cols)
        print("[CHECK] foot_jacobians shape:", tuple(J.shape), " (B,4,3,cols)")
        cols = J.shape[-1]

        # Jacobian 的 DOF 偏移量（浮动基座的话是 6）
        dof_offset = getattr(self, "jac_dof_offset", 0)
        ctrl_cols = self.ctrl_idx_t + dof_offset       # (12,)

        J12 = J[..., ctrl_cols]                       # (B,4,3,12)
        print("[CHECK] J12 shape:", tuple(J12.shape), " (B,4,3,12)")
        print("[CHECK] ||J12||:", float(J12.norm()))

    def _commit_pos_targets(self):
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self.pos_targets)
        )

    def _limit_step(self, tgt_prev: torch.Tensor, tgt_new: torch.Tensor, max_step=0.04):
        # 支持任意形状：逐元素限幅
        delta = torch.clamp(tgt_new - tgt_prev, min=-max_step, max=+max_step)
        return tgt_prev + delta

    @torch.no_grad()
    def contact_force_values(self):
        self.gym.refresh_net_contact_force_tensor(self.sim)
        F = self.net_cf[:, self.feet_local, :].norm(dim=-1, keepdim=True)  # (B,4,1)
        return F

    @torch.no_grad()
    def contact_flags(self, thresh=None):
        # ---- hysteresis contact ----
        cfg = self.cfg
        F = self.contact_force_values().squeeze(-1)  # (B,4)

        on  = float(getattr(cfg, "contact_on_n",  cfg.contact_thresh_n))
        off = float(getattr(cfg, "contact_off_n", cfg.contact_thresh_n * 0.6))

        if not hasattr(self, "_contact_state"):
            self._contact_state = (F > on).float()  # (B,4)

        # 规则：>on 置 1；<off 置 0；中间保持
        self._contact_state = torch.where(F > on,  torch.ones_like(self._contact_state), self._contact_state)
        self._contact_state = torch.where(F < off, torch.zeros_like(self._contact_state), self._contact_state)

        return self._contact_state.unsqueeze(-1)  # (B,4,1)

    # reset - 让所有机器人重新站好
    def reset(self, it: Optional[int] = None):
        dev = self.device
        
        self.t = 0
        # 1) 步态选择：随机 or 固定
        if hasattr(self, "gait_table"):
            if self.cfg.gait_mode < 0:
                # -1: 每个 env 随机 gait
                self.gait_ids = torch.randint(
                    low=0, high=self.num_gaits, size=(self.B,), device=dev
                )
            else:
                # 固定 gait，所有 env 同一个
                gid = int(self.cfg.gait_mode)
                gid = max(0, min(self.num_gaits - 1, gid))
                self.gait_ids = torch.full(
                    (self.B,), gid, dtype=torch.long, device=dev
                )

            # 根据 gait_ids 更新每个 env 的相位偏移
            self.leg_phase_offsets_B = self.gait_table[self.gait_ids]      # (B,4)

        # 2) 随机起始相位 (B,)
        #self.phase = 2 * math.pi * torch.rand(self.B, device=dev)

        # 2) 随机起始相位 (B,) —— 但避免落在“腾空段”
        max_try = 50
        for _ in range(max_try):
            phase_try = 2 * math.pi * torch.rand(self.B, device=dev)      # (B,)
            phases = self.leg_phase_offsets_B + phase_try.view(self.B, 1) # (B,4)
            beta_B, _, _ = self._get_beta_minfeet_allow_aerial()          # (B,)
            stance = (self._phase_u(phases) < beta_B.view(self.B, 1)).float()  # (B,4)
            ok = (stance.sum(dim=1) >= 2)                                 # (B,)
            if ok.all():
                self.phase = phase_try
                break
        else:
            self.phase = phase_try  # 实在找不到就认命（一般不会到这里）
        # 3) 每个 env 随机 yaw
        yaws = torch.empty(self.B, device=dev)
        for i in range(self.B):
            yaws[i] = random.uniform(-math.pi, math.pi)
        self.last_reset_yaw = yaws.clone()

        # root state
        self.gym.refresh_actor_root_state_tensor(self.sim)
        for i in range(self.B):
            root = self.root_state[i]

            ox, oy, oz = self.env_origins[i]
            root[0] = float(ox.item())
            root[1] = float(oy.item())
            root[2] = float(oz.item()) + self.cfg.h0
            q = quat_from_rpy(0.0, 0.0, float(yaws[i].item()))
            root[3] = q.x
            root[4] = q.y
            root[5] = q.z
            root[6] = q.w
            root[7:10] = 0.0
            root[10:13] = 0.0
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_state))

        # 3) 机体前向目标速度 vx_star (B,)
        #vx = 0.2
        #self.vx_star = torch.full((self.B,), vx, device=dev)

        # 3) 高层随机速度指令 cmd_rand = [vx_cmd, vy_cmd, yaw_rate_cmd]
        if self.cfg.rand_cmd:
            vx_rand  = torch.empty(self.B, device=dev).uniform_(self.cfg.vx_min, self.cfg.vx_max)
            vy_rand  = torch.empty(self.B, device=dev).uniform_(self.cfg.vy_min, self.cfg.vy_max)
            yaw_rand = torch.empty(self.B, device=dev).uniform_(self.cfg.yaw_min, self.cfg.yaw_max)
        else:
            vx_rand  = torch.full((self.B,), 1.0,  device=dev)  # 论文里固定 0.2 m/s 的简单任务
            vy_rand  = torch.zeros(self.B, device=dev)
            yaw_rand = torch.zeros(self.B, device=dev)

        self.cmd_rand = torch.stack([vx_rand, vy_rand, yaw_rand], dim=1)  # (B,3)
        self.vx_star  = self.cmd_rand[:, 0]                               # 兼容旧代码（只用 vx）
        
        
        # === 速度相关的步频 / 摆动高度（per-env） ===
        # step frequency (Hz): for Example 3.3, sample uniformly in [step_freq_min, step_freq_max] on reset
        if getattr(self.cfg, "rand_step_freq", False):
            self.step_freq_B = torch.empty(self.B, device=dev).uniform_(self.cfg.step_freq_min, self.cfg.step_freq_max)
        else:
            self.step_freq_B = torch.full((self.B,), self.cfg.step_freq, device=dev)

        # keep swing height as a constant by default (you can also make it depend on vx_star if you want)
        self.swing_height_B = torch.full((self.B,), self.cfg.swing_height, device=dev)

        # 4) 关节目标回到默认姿态
        base_local = torch.as_tensor(self.q_default_full, device=dev, dtype=torch.float32)
        self.local_targets = base_local.view(1, -1).repeat(self.B, 1)
        self.pos_targets_batch[:] = self.local_targets
        self._commit_pos_targets()

        for _ in range(self.cfg.settle_steps_reset):
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

        # 5) 工程版：给一点初始前向速度
        if not PURE_PAPER_MODE:
            self.gym.refresh_actor_root_state_tensor(self.sim)
            for i in range(self.B):
                root = self.root_state[i]
                v0_body = 0.10
                if hasattr(self, "stop_cmd_mask") and bool(self.stop_cmd_mask[i].item()):
                    v0_body = 0.0
                yaw = float(self.last_reset_yaw[i].item())
                vx_world = v0_body * math.cos(yaw)
                vy_world = v0_body * math.sin(yaw)
                root[7] = vx_world
                root[8] = vy_world
                root[9] = 0.0
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_state))

        # 6) 刷新缓存 & SRBD 初始化
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        # 清所有 env 的 DOF：q=默认，qd=0
        base_q = torch.as_tensor(self.q_default_full, device=dev, dtype=torch.float32)  # (dof_count,)
        self.dof_state_view[:, :, 0] = base_q.view(1, -1).repeat(self.B, 1)
        self.dof_state_view[:, :, 1] = 0.0

        # 写回所有 actor（indexed 需要 int32）
        actor_ids = self.actor_indices_t.to(dtype=torch.int32)  # (B,)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state_t),
            gymtorch.unwrap_tensor(actor_ids),
            actor_ids.numel()
       )
 
        
        self._update_cache()
        self._srbd_init_from_isaac()

        self.srbd_p = self.srbd_p.detach()
        self.srbd_v = self.srbd_v.detach()
        self.srbd_q = self.srbd_q.detach()
        self.srbd_w = self.srbd_w.detach()
 
        # reset 后：重置 last_contact_*，避免 stance 锁到“旧 episode”的脚位置
        with torch.no_grad():
            p_foot = self.foot_positions()            # (B,4,3)
            self.last_contact_xy[:] = p_foot[..., 0:2]
            self.last_contact_z[:]  = p_foot[..., 2]
            # reset 时：liftoff 起点也重置为当前脚（避免上一 episode 残留）
            self.last_liftoff_xyz[:] = p_foot
            # prev stance 置 1，避免 reset 后第一帧被误判为“刚离地”
            self.prev_stance_mask[:] = 1.0
            # contact edge cache 同步
            if not hasattr(self, "prev_contact_flags"):
                self.prev_contact_flags = torch.zeros(self.B, 4, 1, device=self.device)
            self.prev_contact_flags[:] = self.contact_flags()
        
        # env0 stride debug cache reset
        
        self._stride_have_td0[:] = False
        self._stride_count0[:] = 0
        self._stride_last_td_step0[:] = -10000
        # 关键：last_td_xy0 也要重置成当前脚位置，避免第一次 stride 爆表
        p_foot0 = self.foot_positions()[0, :, 0:2].detach()
        self._stride_last_td_xy0[:] = p_foot0
    
    # -------- 新增：只重置部分 env 的局部 reset --------
    def reset_envs(self, env_ids):
        """
        局部 reset：只重置 env_ids 这几条狗到
        “随机化但可控的初始姿态 + 随机速度指令”。
        env_ids: 可以是 int / list[int] / numpy / torch.Tensor
        """
        dev = self.device

        # 统一成 1D LongTensor
        if not torch.is_tensor(env_ids):
            env_ids = torch.as_tensor(env_ids, device=dev, dtype=torch.long)
        else:
            env_ids = env_ids.to(device=dev, dtype=torch.long)
        if env_ids.numel() == 0:
            return
        env_ids = env_ids.view(-1)

        # 1) 为这些 env 选择步态（gait）
        if hasattr(self, "gait_table"):
            if self.cfg.gait_mode < 0:
                # -1: 每个 env 随机一个 gait
                new_gids = torch.randint(
                    low=0, high=self.num_gaits,
                    size=(env_ids.numel(),),
                    device=dev
                )
            else:
                gid = int(self.cfg.gait_mode)
                gid = max(0, min(self.num_gaits - 1, gid))
                new_gids = torch.full(
                    (env_ids.numel(),),
                    gid,
                    dtype=torch.long,
                    device=dev
                )
            # 更新这几条狗的 gait_id 和相位偏移
            self.gait_ids[env_ids] = new_gids
            self.leg_phase_offsets_B[env_ids] = self.gait_table[new_gids]

        # 2) 这几条狗随机一个起始相位
        #self.phase[env_ids] = 2 * math.pi * torch.rand(env_ids.numel(), device=dev)

        # 2) 这几条狗随机一个起始相位 —— 但避免落在“腾空段”
        n = env_ids.numel()
        max_try = 50
        for _ in range(max_try):
            phase_try = 2 * math.pi * torch.rand(n, device=dev)  # (n,)
            # 取这批 env 的相位偏移
            phase_offsets = self.leg_phase_offsets_B[env_ids]    # (n,4)
            phases = phase_offsets + phase_try.view(n, 1)        # (n,4)
            # 取这批 env 的 beta（注意：_get_beta... 是按全 B 生成的，所以取子集）
            beta_B, _, _ = self._get_beta_minfeet_allow_aerial() # (B,)
            beta_sub = beta_B[env_ids]                            # (n,)

            stance = (self._phase_u(phases) < beta_sub.view(n, 1)).float()  # (n,4)
            ok = (stance.sum(dim=1) >= 2)                                   # (n,)
            if ok.all():
                self.phase[env_ids] = phase_try
                break
        else:
            self.phase[env_ids] = phase_try
        # 3) 为这几条狗随机 yaw & 根状态 (base pose + vel)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        # 随机 yaw
        yaws = torch.empty(env_ids.numel(), device=dev)
        for k in range(env_ids.numel()):
            yaws[k] = random.uniform(-math.pi, math.pi)
        self.last_reset_yaw[env_ids] = yaws.clone()

        # 写回 root_state
        for k, env_id in enumerate(env_ids):
            i = int(env_id.item())
            root = self.root_state[i]
            ox, oy, oz = self.env_origins[i]
            root[0] = float(ox.item())
            root[1] = float(oy.item())
            root[2] = float(oz.item()) + self.cfg.h0
            q = quat_from_rpy(0.0, 0.0, float(yaws[k].item()))
            root[3] = q.x
            root[4] = q.y
            root[5] = q.z
            root[6] = q.w
            # 线速度 / 角速度清零
            root[7:10]  = 0.0
            root[10:13] = 0.0

        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.root_state)
        )

        # 4) 为这几条狗重置 / 随机速度指令 cmd_rand
        if not hasattr(self, "cmd_rand"):
            self.cmd_rand = torch.zeros(self.B, 3, device=dev)
        if not hasattr(self, "vx_star"):
            self.vx_star = torch.zeros(self.B, device=dev)

        if self.cfg.rand_cmd:
            vx_rand  = torch.empty(env_ids.numel(), device=dev).uniform_(self.cfg.vx_min,  self.cfg.vx_max)
            vy_rand  = torch.empty(env_ids.numel(), device=dev).uniform_(self.cfg.vy_min,  self.cfg.vy_max)
            yaw_rand = torch.empty(env_ids.numel(), device=dev).uniform_(self.cfg.yaw_min, self.cfg.yaw_max)
        else:
            vx_rand  = torch.full((env_ids.numel(),), 1.0, device=dev)
            vy_rand  = torch.zeros(env_ids.numel(), device=dev)
            yaw_rand = torch.zeros(env_ids.numel(), device=dev)

        self.cmd_rand[env_ids, 0] = vx_rand
        self.cmd_rand[env_ids, 1] = vy_rand
        self.cmd_rand[env_ids, 2] = yaw_rand
        self.vx_star[env_ids]     = vx_rand      # 仍然给 Raibert / loss 用

        # 5) 对应的步频 / 摆动高度恢复成默认值
        if not hasattr(self, "step_freq_B"):
            self.step_freq_B = torch.full((self.B,), self.cfg.step_freq, device=dev)
        if not hasattr(self, "swing_height_B"):
            self.swing_height_B = torch.full((self.B,), self.cfg.swing_height, device=dev)

        # step frequency: either resample or keep constant
        if getattr(self.cfg, "rand_step_freq", False):
            self.step_freq_B[env_ids] = torch.empty(env_ids.numel(), device=dev).uniform_(self.cfg.step_freq_min, self.cfg.step_freq_max)
        else:
            self.step_freq_B[env_ids] = self.cfg.step_freq

        # swing height back to default
        self.swing_height_B[env_ids] = self.cfg.swing_height

        # 6) 把这几条狗的关节目标回到默认站立
        base_local = torch.as_tensor(self.q_default_full, device=dev, dtype=torch.float32)
        # local_targets: (B, dof_count)
        self.local_targets[env_ids]      = base_local.unsqueeze(0)
        self.pos_targets_batch[env_ids]  = self.local_targets[env_ids]
        self._commit_pos_targets()

        # 7) 刷新 cache & SRBD（这里简单起见，全局刷一次就行）
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        
        for env_id in env_ids.tolist():
            self.dof_state_view[env_id, :, 0] = torch.as_tensor(self.q_default_full, device=dev)
            self.dof_state_view[env_id, :, 1] = 0.0

        actor_ids = self.actor_indices_t[env_ids].to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state_t),
            gymtorch.unwrap_tensor(actor_ids),
            actor_ids.numel()
        )

        self._update_cache()
        self._srbd_init_from_isaac()

        self.srbd_p = self.srbd_p.detach()
        self.srbd_v = self.srbd_v.detach()
        self.srbd_q = self.srbd_q.detach()
        self.srbd_w = self.srbd_w.detach()

        # 局部 reset 后：只更新这些 env 的 last_contact_*（否则 stance 会锁到旧脚印）
        with torch.no_grad():
            p_foot = self.foot_positions()            # (B,4,3)
            self.last_contact_xy[env_ids] = p_foot[env_ids, :, 0:2]
            self.last_contact_z[env_ids]  = p_foot[env_ids, :, 2]
            # 同步重置 liftoff 起点
            self.last_liftoff_xyz[env_ids] = p_foot[env_ids]
            self.prev_stance_mask[env_ids] = 1.0
            # 同步 contact edge cache
            if not hasattr(self, "prev_contact_flags"):
                self.prev_contact_flags = torch.zeros(self.B, 4, 1, device=self.device)
            self.prev_contact_flags[env_ids] = self.contact_flags()[env_ids]
        
        if (env_ids == 0).any():
            self._stride_have_td0[:] = False
            self._stride_count0[:] = 0
            self._stride_last_td_step0[:] = -10000
            # 关键：last_td_xy0 也要重置成当前脚位置，避免第一次 stride 爆表
            p_foot0 = self.foot_positions()[0, :, 0:2].detach()
            self._stride_last_td_xy0[:] = p_foot0

    #此公式专门用来推导 步频 以及 抬腿高度 -> 通过 速度指令 以及 "固定步长" 来更新步频 f 
    def _update_gait_from_cmd(self):
        """
        固定步长 + 频率跟随速度指令（批量 B 版）
        - 用 cmd_rand[:,0:2] 的速度模长作为 |v_cmd|
        - 对每个 gait 设一个固定 hip->landing 距离 L_land（m）
        - 通过 Raibert 公式里的 delta = |v|/(4f) 反推 f = |v|/(4*L_land)
        - clamp 到 [step_freq_min, step_freq_max]，并做 deadzone 与 rough terrain 缩放
        """
        if not getattr(self.cfg, "step_freq_from_cmd", False):
            return

        dev = self.device
        B = self.B
        cfg = self.cfg

        # -------- 速度命令（你已经维护了 cmd_rand / vx_star）--------
        if not hasattr(self, "cmd_rand"):
            # 兜底：只用 vx_star
            v_cmd_xy = torch.stack([self.vx_star, torch.zeros_like(self.vx_star)], dim=-1)
        else:
            v_cmd_xy = self.cmd_rand[:, 0:2]  # (B,2) in body frame (your convention)
        v_mag = torch.linalg.norm(v_cmd_xy, dim=1)  # (B,)
        v_hi = max(0.20, float(getattr(cfg, "vx_max", 0.5)))  # 用于归一化

        # -------- gait_id (B,) --------
        gait_ids = getattr(self, "gait_ids", torch.ones(B, dtype=torch.long, device=dev)).to(dev)

        # -------- deadzone --------
        dead = float(getattr(cfg, "cmd_deadzone", 0.05))  # m/s
        yaw_dead = float(getattr(cfg, "yaw_deadzone", dead))

        if hasattr(self, "cmd_rand"):
            v_mag = torch.linalg.norm(self.cmd_rand[:, 0:2], dim=1)   # (B,)
            yaw_mag = torch.abs(self.cmd_rand[:, 2])                  # (B,)
        else:
            v_mag = torch.abs(self.vx_star)
            yaw_mag = torch.zeros_like(v_mag)

        # 只要 “速度 or 转向” 有一个显著，就认为需要迈步/推进相位
        self.move_mask_B = ((v_mag >= dead) | (yaw_mag >= yaw_dead)).float()   # (B,)

        # -------- 固定步长表：这里的 L_land 是 “hip 到落脚点” 的前向距离（m）--------
        # 你可以按自己观感改；这些值在 0.06~0.09m 比较好用（对应 stride 约 2*L_land 的量级）
        L_table = torch.tensor(
            #[0.0, 0.065, 0.065, 0.075, 0.085],  # stand, trot, pace, bound, gallop
            #[0.0, 0.080, 0.065, 0.075, 0.085],  # stand, trot, pace, bound, gallop
            #[0.0, 0.10, 0.065, 0.075, 0.085],
            #[0.0, 0.12, 0.020, 0.075, 0.085],
            #[0.0, 0.08, 0.080, 0.075, 0.075],
            #[0.0, 0.08, 0.12, 0.075, 0.075],   #0.12

            #[0.0, 0.125, 0.065, 0.055, 0.085],   # 已经调好的版本
            [0.0, 0.08, 0.065, 0.055, 0.085],   #0.12
            
            
            
            dtype=torch.float32, device=dev
        )
        L_land = L_table[gait_ids].clamp(min=1e-3)  # (B,)

        # -------- rough terrain 缩放（保守一点：更小步、略低频、更高抬腿）--------
        rough = 1.0 if bool(globals().get("USE_COMPLEX_TERRAIN", False)) else 0.0
        L_land = L_land * (1.0 - 0.10 * rough)               # 步长更短一点
        freq_scale = (1.0 - 0.15 * rough)                    # 频率更保守一点（更低）
        #height_scale = (1.0 + 0.40 * rough)                  # 抬腿更高一点
        height_scale = 1.0                 # 抬腿更高一点

        # -------- 由固定步长反推 f：delta ≈ |v|/(4f)  =>  f ≈ |v|/(4*L_land) --------
        f_raw = (v_mag / (4.0 * L_land + 1e-6)) * freq_scale

        fmin = float(cfg.step_freq_min)
        fmax = float(cfg.step_freq_max)
        f = torch.clamp(f_raw, fmin, fmax)

        # deadzone：低速时别硬追“固定步长”，直接给最小频率 + 很小抬腿
        f = torch.where(v_mag < dead, torch.full((B,), fmin, device=dev), f)
        self.step_freq_B = f

        # -------- swing height：随速度上升 + rough 增益 + deadzone 降低 --------
        h0 = float(cfg.swing_height)
        #h_max = float(getattr(cfg, "swing_height_max", 0.11))
        #h_max = float(getattr(cfg, "swing_height_max", 0.015))

        # ✅ 修复：确保 h_max >= h0，且默认给个合理值（0.05）
        h_max = float(getattr(cfg, "swing_height_max", 0.05))
        h_max = max(h_max, h0)
        s = torch.clamp(v_mag / (v_hi + 1e-6), 0.0, 1.0)
        h = (h0 + (h_max - h0) * s) * height_scale
        h = torch.where(v_mag < dead, torch.full((B,), 0.5 * h0, device=dev), h)
        self.swing_height_B = h


    @torch.no_grad()
    def demo_trot(self, seconds=6.0, amp_thigh=0.25, amp_calf=0.45):
        """简单 demo：所有 env 同样的 trot"""
        steps = int(seconds / self.cfg.dt)
        self.phase = torch.zeros(self.B, device=self.device)
        for _ in range(steps):
            phases = self.leg_phase_offsets + self.phase[0]
            swing  = torch.clamp(torch.sin(phases), min=0.0)
            q_ref  = torch.as_tensor(self.q_default_np, device=self.device).clone()
            q_ref[1::3] += amp_thigh * swing
            q_ref[2::3] -= amp_calf  * swing

            target_slice = self.local_targets.clone()
            target_slice[:, self.ctrl_idx_t] = q_ref * self.ctrl_sign
            target_slice = torch.max(torch.min(target_slice, self._hi_slice), self._lo_slice)
            target_slice = self._limit_step(self.local_targets, target_slice, max_step=0.08)
            self.local_targets = target_slice
            self.pos_targets_batch[:] = self.local_targets
            self._commit_pos_targets()

            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)

            if self.viewer is not None:
                self.gym.step_graphics(self.sim)
                self._update_chase_camera()
                self.gym.draw_viewer(self.viewer, self.sim, True)
                self.gym.sync_frame_time(self.sim)

            self.phase = (self.phase + 2*math.pi*self.cfg.step_freq*self.cfg.dt) % (2*math.pi)

    # 四元数 -> 欧拉角（batch 版）
    def _rpy_from_quat_wxyz(self):
        qw, qx, qy, qz = self.base_quat[:, 0], self.base_quat[:, 1], self.base_quat[:, 2], self.base_quat[:, 3]

        sinr_cosp = 2.0 * (qw*qx + qy*qz)
        cosr_cosp = 1.0 - 2.0 * (qx*qx + qy*qy)
        roll = torch.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (qw*qy - qz*qx)
        sinp = torch.clamp(sinp, -1.0 + 1e-6, 1.0 - 1e-6)
        pitch = torch.asin(sinp)

        siny_cosp = 2.0 * (qw*qz + qx*qy)
        cosy_cosp = 1.0 - 2.0 * (qy*qy + qz*qz)
        yaw = torch.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def _update_cache(self):
        """
        从 Isaac 刷新 batch base 位姿 / 速度等。
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        base = self.root_state  # (B,13)

        self.base_pos  = base[:, 0:3]                      # (B,3)
        q_xyzw = base[:, 3:7]                              # (B,4)
        self.base_quat = torch.stack(                      # (B,4) wxyz
            [q_xyzw[:, 3], q_xyzw[:, 0], q_xyzw[:, 1], q_xyzw[:, 2]], dim=1
        )

        # 世界系速度 / 角速度
        self.base_lin_world = base[:, 7:10]                # (B,3)
        self.base_ang_world = base[:, 10:13]               # (B,3)

        # 机体系速度 / 角速度
        base_lin_body = []
        base_ang_body = []
        for i in range(self.B):
            v_b = quat_rotate_inverse_wxyz(self.base_quat[i], self.base_lin_world[i], self.device)
            w_b = quat_rotate_inverse_wxyz(self.base_quat[i], self.base_ang_world[i], self.device)
            base_lin_body.append(v_b)
            base_ang_body.append(w_b)
        self.base_lin_body = torch.stack(base_lin_body, dim=0)  # (B,3)
        self.base_ang_body = torch.stack(base_ang_body, dim=0)  # (B,3)

        # world frame 别名
        self.base_lin = self.base_lin_world
        self.base_ang = self.base_ang_world

        # 关节
        dof = self.dof_state_view  # (B,dof_count,2)
        self.q  = dof[..., 0]      # (B,dof_count)
        self.qd = dof[..., 1]      # (B,dof_count)

        # SRBD 2D 旧变量（保持接口）
        self.p = torch.stack([self.base_pos[:, 0], self.base_pos[:, 2]], dim=1)  # (B,2)
        self.v = torch.stack([self.base_lin[:, 0], self.base_lin[:, 2]], dim=1)  # (B,2)

        self.roll, self.pitch, self.yaw = self._rpy_from_quat_wxyz()  # (B,)
        self.theta = self.pitch.clone()
        self.omega = self.base_ang[:, 1]  # (B,)


    # 摄像机跟随第 0 只狗
    def _yaw_from_quat(self) -> float:
        qw, qx, qy, qz = self.base_quat[0]
        return math.atan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy*qy + qz*qz))

    def _update_chase_camera(self):
        if self.viewer is None or not self.cam_follow: return
        px, py, pz = [float(v) for v in self.base_pos[0, :3]]
        yaw = self._yaw_from_quat()
        back = np.array([-math.cos(yaw), -math.sin(yaw), 0.0])
        up   = np.array([0.0, 0.0, 1.0])
        desired_eye = np.array([px, py, pz]) + self.cam_dist * back + self.cam_height * up
        desired_tgt = np.array([px, py, pz + 0.30])
        if self._cam_eye is None:
            self._cam_eye = desired_eye; self._cam_tgt = desired_tgt
        else:
            a = float(self.cam_smooth)
            self._cam_eye = (1 - a) * self._cam_eye + a * desired_eye
            self._cam_tgt = (1 - a) * self._cam_tgt + a * desired_tgt
        self.gym.viewer_camera_look_at(
            self.viewer, None,
            gymapi.Vec3(*self._cam_eye.tolist()),
            gymapi.Vec3(*self._cam_tgt.tolist()),
        )

    # ---------------- helpers / sensors ----------------
    @torch.no_grad()
    def foot_positions(self):
        """足端位置（世界系），(B,4,3)"""
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        pos = self.rb_state_t[:, self.feet_local, 0:3]  # (B,4,3)
        return pos

    @torch.no_grad()
    def foot_jacobians(self):
        """
        足端雅可比矩阵，返回 (B, 4, 3, cols)

        注意：
        - 浮动基座时 Jacobian 形状是 (B, num_links, 6, num_dofs+6)
          前 6 列对应 base DOF，我们要用的是后面 num_dofs 那部分。
        """
        self.gym.refresh_jacobian_tensors(self.sim)
        J_flat = self.jacobian                    # 原始 tensor

        B_j, nb_j, six, cols = J_flat.shape
        assert B_j == self.B
        assert six == 6, "Jacobian 3rd dim 必须为 6（线+角速度）"

        # 第一次调用时，根据列数自动确定 DOF 偏移量
        if not hasattr(self, "jac_dof_offset"):
            if cols == self.dof_count + 6:
                # 浮动基座：前 6 列是 base DOF
                self.jac_dof_offset = 6
                print(f"[INFO] Jacobian detected as floating-base: cols={cols}, dof_count={self.dof_count}, offset=6")
            elif cols == self.dof_count:
                # 固定基座：刚好等于关节 DOF 数
                self.jac_dof_offset = 0
                print(f"[INFO] Jacobian detected as fixed-base: cols={cols}, dof_count={self.dof_count}, offset=0")
            else:
                # 非标准情况：把最后 self.dof_count 列当成关节 DOF
                self.jac_dof_offset = max(0, cols - self.dof_count)
                print(f"[WARN] Unexpected Jacobian shape {J_flat.shape}, "
                      f"treating last {self.dof_count} columns as joint DOFs, "
                      f"offset={self.jac_dof_offset}")

        # 只取线速度部分 0:3
        J_lin = J_flat[:, :, 0:3, :]              # (B, nb_j, 3, cols)

        # 取四个足端刚体对应的行
        assert nb_j > max(self.feet_local), "Jacobian num_links < feet_local index"
        J_feet = torch.stack(
            [J_lin[:, self.feet_local[i]] for i in range(4)],
            dim=1
        )                                         # (B, 4, 3, cols)

        return J_feet


    # ---------------- swing trajectory ----------------
    # 二次抛物线插值
    def _swing_parabola(self, p0, pm, p1, s):
        c = p0
        b = 4*(pm - (p0 + p1)/2.0)
        a = p1 - p0 - b
        return a*(s**2) + b*s + c

    # 生成摆动腿参考轨迹 
    def swing_ref_traj(self, phases, p_foot_now):
        """
        严格 duty-factor 版摆动轨迹（batch）
        phases: (B,4)
        p_foot_now: (B,4,3)
        return pref: (B,4,3)
        """
        dev = self.device
        B = self.B

        # per-env duty factor β & T_stance/T_swing
        beta_B, _, _ = self._get_beta_minfeet_allow_aerial()           # (B,)
        step_freq = getattr(self, "step_freq_B", torch.full((B,), self.cfg.step_freq, device=dev))
        T = 1.0 / step_freq                                            # (B,)
        T_stance = beta_B * T                                          # (B,)
        # Raibert 落脚点（用 T_stance）
        p_hip_xy, p_land_xy = self._raibert_footholds_body(T_stance)    # (B,4,2)

        # swing phase in [0,1] only when u>=beta
        u = self._phase_u(phases)                                      # (B,4)
        beta = beta_B.view(B, 1)
        is_swing = (u >= beta)

        swing_phase = torch.zeros_like(u)
        swing_phase[is_swing] = (u[is_swing] - beta[is_swing]) / (1.0 - beta[is_swing] + 1e-6)
        s = swing_phase.clamp(0.0, 1.0).unsqueeze(-1)                   # (B,4,1)
        # xz parabola
        p0 = p_foot_now[..., [0, 2]]                                    # (B,4,2)

        # per-env swing height
        if hasattr(self, "swing_height_B"):
            h = self.swing_height_B.view(B, 1).expand(-1, 4)            # (B,4)
            pm = torch.stack([p0[..., 0], p0[..., 1] + h], dim=-1)      # (B,4,2)
        else:
            pm = torch.stack([p0[..., 0], p0[..., 1] + self.cfg.swing_height], dim=-1)
        z_land = self.last_contact_z                                    # (B,4)
        p1 = torch.stack([p_land_xy[..., 0], z_land], dim=-1)           # (B,4,2)

        pref_xz = self._swing_parabola(p0, pm, p1, s)                   # (B,4,2)

        pref = torch.zeros_like(p_foot_now)
        pref[..., 0] = pref_xz[..., 0]
        pref[..., 1] = p_hip_xy[..., 1]
        pref[..., 2] = pref_xz[..., 1]
        return pref
    
    def _update_foot_targets_from_command(self, phases, p_foot_now, return_vref: bool = False):
        """
        严格图 9.2/9.3：stance/swing 由 duty factor β 定义；running/bound/gallop 允许腾空段
        """
        dev, cfg, B = self.device, self.cfg, self.B

        # ---------- 0) contact（只用于 last_contact_z 等；mask 默认不再用 contact 掺混） ----------
        contact = self.contact_flags(thresh=cfg.contact_thresh_n)  # (B,4,1)

        # ---------- 1) 由 gait 得到 duty factor β & min_feet ----------
        beta_B, min_feet_B, allow_aerial = self._get_beta_minfeet_allow_aerial()

        # ---------- 2) phase-based stance_mask（严格定义） ----------
        stance_mask = self._mix_stance(
            phases=phases,
            contact_flags=contact,
            beta_B=beta_B,
            min_feet_B=min_feet_B,
            w_phase=1.0,
            w_contact=0.0
        )  # (B,4,1)
        swing_mask = 1.0 - stance_mask

        # ---------- 2.5) 记录离地瞬间 liftoff 起点 (x0,y0,z0) ----------
        # liftoff: 上一帧是 stance，这一帧变成 swing
        if not hasattr(self, "prev_stance_mask"):
            self.prev_stance_mask = torch.ones_like(stance_mask)
        # 重要：rollout 多步再 backward 时，任何“状态缓存”都必须 detach + clone，
        # 且避免 in-place 写入，否则 autograd 会报 version mismatch
        with torch.no_grad():
            liftoff = (self.prev_stance_mask > 0.5) & (stance_mask < 0.5)          # (B,4,1)
            liftoff3 = liftoff.expand(-1, -1, 3)                                   # (B,4,3)
            self.last_liftoff_xyz = torch.where(
                liftoff3,
                p_foot_now.detach(),                                               # ★ detach
                self.last_liftoff_xyz
            ).clone()                                                               # ★ clone (断开存储别名)


        # ---------- 3) 取高层命令 (vx, vy, yaw_rate) ----------
        if hasattr(self, "cmd_rand"):
            v_cmd_body_xy = self.cmd_rand[:, 0:2]   # (B,2)
            yaw_rate_cmd  = self.cmd_rand[:, 2]     # (B,)
        else:
            v_cmd_body_xy = torch.stack([self.vx_star, torch.zeros_like(self.vx_star)], dim=-1)
            yaw_rate_cmd  = torch.zeros(B, device=dev)

        # ---------- stop：强制全足 stance ----------
        dead = float(getattr(cfg, "cmd_deadzone", 0.05))
        v_mag = torch.linalg.norm(v_cmd_body_xy, dim=1)          # (B,)
        stop_env = (v_mag < dead)                                # (B,)
        stop3 = stop_env.view(B, 1, 1)
        v_cmd_body_xy = torch.where(stop_env[:, None], torch.zeros_like(v_cmd_body_xy), v_cmd_body_xy)
        yaw_rate_cmd  = torch.where(stop_env, torch.zeros_like(yaw_rate_cmd), yaw_rate_cmd)

        stance_mask = torch.where(stop3, torch.ones_like(stance_mask), stance_mask)
        swing_mask  = 1.0 - stance_mask

        # ---------- 4) body->world yaw 旋转 ----------
        yaw = self.yaw
        cy = torch.cos(yaw); sy = torch.sin(yaw)
        R_yaw = torch.stack(
            [torch.stack([cy, -sy], dim=-1),
             torch.stack([sy,  cy], dim=-1)], dim=1
        )  # (B,2,2)

        v_cmd_world_xy = torch.einsum("bij,bj->bi", R_yaw, v_cmd_body_xy)  # (B,2)

        # ---------- 5) STANCE 分支（符合图9.2：支撑脚世界系静止） ----------
        # 直接锁死在最近一次“触地瞬间”的落脚位置（world frame）
        # 这样 PD target 不会拉着支撑脚“往后拖”，可显著减少打滑/前倾
        p_stance = p_foot_now.clone()
        p_stance[..., 0:2] = self.last_contact_xy
        p_stance[..., 2]   = self.last_contact_z


        # ---------- 6) SWING 分支：摆线轨迹 (教材 9.3 / 式 9.14-9.15) ----------
        dt = cfg.dt
        step_freq = getattr(self, "step_freq_B", torch.full((B,), cfg.step_freq, device=dev))
        T = 1.0 / step_freq                                # (B,)
        # duty factor
        T_stance = beta_B * T                              # (B,)
        T_swing  = (1.0 - beta_B) * T                      # (B,)

        # Raibert touchdown（严格对齐图 9.6/9.7：含 (1-p)T_swing 与 +k(v-vd)）
        _, p_land_xy_world = self._raibert_touchdown_world(phases)   # (B,4,2)

        u = self._phase_u(phases)                          # (B,4)
        beta = beta_B.view(B, 1)
        is_swing = (u >= beta)

        swing_phase = torch.zeros_like(u)
        #swing_phase[is_swing] = (u[is_swing] - beta[is_swing]) / (1.0 - beta[is_swing] + 1e-6)

        # u: (B,4) in [0,1)
        u = self._phase_u(phases)  # (B,4)

        beta_B, min_feet_B, allow_aerial = self._get_beta_minfeet_allow_aerial()  # beta_B: (B,) or (B,1)

        B = u.shape[0]
        beta4 = beta_B.view(B, 1).expand(B, 4)  # ✅ 强制变成 (B,4)，和 u 对齐

        # swing 区间：u in [beta, 1)
        is_swing = (u >= beta4)

        den = (1.0 - beta4).clamp_min(1e-6)
        raw = ((u - beta4) / den).clamp(0.0, 1.0)  # (B,4)
        # ✅ 不要用 swing_phase[mask] = ... 这种赋值；直接 where
        swing_phase = torch.where(is_swing, raw, torch.zeros_like(u))  # (B,4)
        swing_phase = swing_phase.clamp(0.0, 1.0)          # (B,4)

        # p in [0,1] 是教材里的相位参数（式 9.14/9.15）
        p = swing_phase.clamp(0.0, 1.0)                                      # (B,4)
        theta = 2.0 * math.pi * p                                            # (B,4)
        # s(p) = (2πp - sin 2πp)/(2π)，端点速度为 0
        s = (theta - torch.sin(theta)) / (2.0 * math.pi)                     # (B,4)

        # 起点 p0 = liftoff（世界系）
        # 重要：读出来也要 detach+clone，避免后续状态更新影响到本步前向保存的值
        p0_xy = self.last_liftoff_xyz[..., 0:2].detach().clone()              # (B,4,2)
        p0_z  = self.last_liftoff_xyz[..., 2].detach().clone()                # (B,4)
        # 终点 p1 = touchdown（世界系）
        p1_xy = p_land_xy_world                                               # (B,4,2)
        # 终点高度：保持你原先的“落到最近接触高度”（粗糙地形也更稳）
        p1_z  = self.last_contact_z                                           # (B,4)

        # XY：摆线插值（式 9.14 的 x/y 形式）
        foot_xy_swing = p0_xy + (p1_xy - p0_xy) * s.unsqueeze(-1)             # (B,4,2)

        # Z：更通用的“线性 + cos bump”
        # z(p)=z0 + (z1-z0)p + h/2*(1-cos 2πp)
        h_env = getattr(self, "swing_height_B", torch.full((B,), cfg.swing_height, device=dev))  # (B,)
        h_leg = h_env.view(B, 1).expand(-1, 4)                                # (B,4)
        bump = 0.5 * h_leg * (1.0 - torch.cos(theta))                         # (B,4)
        foot_z_swing = p0_z + (p1_z - p0_z) * p + bump                         # (B,4)
        p_swing = torch.zeros_like(p_foot_now)
        p_swing[..., 0:2] = foot_xy_swing
        p_swing[..., 2]   = foot_z_swing

        # ====== (新增) 严格教材式摆线速度输出：式 (9.15) ======
        # xdot = (x1-x0)/T * (1 - cos 2πp)
        # ydot = (y1-y0)/T * (1 - cos 2πp)
        # zdot = (z1-z0)/T + (πh/T) * sin 2πp
        # 其中 T 用 swing 时长 T_swing（每条腿同 env 相同；足够贴教材）
        Ts = T_swing.view(B, 1).clamp_min(1e-3)                                # (B,1)
        one_minus_cos = (1.0 - torch.cos(theta))                               # (B,4)
        sin_theta = torch.sin(theta)                                           # (B,4)

        v_xy_ref = (p1_xy - p0_xy) / Ts.unsqueeze(-1)                          # (B,4,2)
        v_xy_ref = v_xy_ref * one_minus_cos.unsqueeze(-1)                      # (B,4,2)

        v_z_ref = (p1_z - p0_z) / Ts                                            # (B,4)
        v_z_ref = v_z_ref + (math.pi * h_leg / Ts) * sin_theta                 # (B,4)

        v_foot_ref_world = torch.zeros_like(p_foot_now)                        # (B,4,3)
        v_foot_ref_world[..., 0:2] = v_xy_ref
        v_foot_ref_world[..., 2]   = v_z_ref

        # stance / stop 时，期望足端速度为 0
        v_foot_ref_world = v_foot_ref_world * swing_mask
        v_foot_ref_world = torch.where(stop3.expand(-1, -1, 3),
                                       torch.zeros_like(v_foot_ref_world),
                                       v_foot_ref_world)

        # ---------- 7) 混合 ----------
        p_foot_target = stance_mask * p_stance + (1.0 - stance_mask) * p_swing

        # stop env 锁定足端
        p_foot_target = torch.where(stop3, p_foot_now, p_foot_target)
        stance_mask   = torch.where(stop3, torch.ones_like(stance_mask), stance_mask)

        # ---------- 8) 更新 prev_stance_mask（用于下一帧检测 liftoff） ----------
        with torch.no_grad():
            self.prev_stance_mask = stance_mask.detach().clone()              # ★ clone 避免别名/版本号问题

        if return_vref:
            return p_foot_target, stance_mask, v_foot_ref_world
        return p_foot_target, stance_mask
    
    def _raibert_touchdown_world(self, phases: torch.Tensor):
        """
        严格对齐教材/图 9.6/9.7 的 touchdown：
            p_land = p_hip(p) + v_now*(1-p)*T_swing + 0.5*T_stance*v_des + k*(v_now - v_des) + x_bias*fwd
        - p (=swing_phase) 是每条腿自己的摆动进度
        - v_now 用当前估计速度（world），v_des 来自 cmd（body->world）
        返回:
            p_hip_xy_world  : (B,4,2)
            p_land_xy_world : (B,4,2)
        """
        dev, cfg, B = self.device, self.cfg, self.B

        beta_B, _, _ = self._get_beta_minfeet_allow_aerial()   # (B,)
        step_freq = getattr(self, "step_freq_B", torch.full((B,), cfg.step_freq, device=dev))
        T = 1.0 / step_freq                                    # (B,)
        T_stance = beta_B * T                                   # (B,)
        T_swing  = (1.0 - beta_B) * T                            # (B,)

        # u in [0,1), p in [0,1]
        u = self._phase_u(phases)                                # (B,4)
        beta4 = beta_B.view(B, 1).expand(B, 4)                   # (B,4)
        den = (1.0 - beta4).clamp_min(1e-6)
        p = ((u - beta4) / den).clamp(0.0, 1.0)                  # (B,4)
        T_left = ((1.0 - p) * T_swing.view(B, 1)).clamp_min(1e-3) # (B,4)

        # yaw -> body->world
        yaw = self.yaw                                           # (B,)
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        R_yaw = torch.stack(
            [torch.stack([cy, -sy], dim=-1),
             torch.stack([sy,  cy], dim=-1)], dim=1
        )                                                        # (B,2,2)

        # hip offsets in body frame
        hip_offsets_body = torch.tensor([
            [ +0.1934, +0.1420 ],
            [ +0.1934, -0.1420 ],
            [ -0.1934, +0.1420 ],
            [ -0.1934, -0.1420 ],
        ], dtype=torch.float32, device=dev).view(1,4,2).expand(B,4,2)

        base_xy = self.base_pos[:, 0:2]                          # (B,2)
        p_hip_xy_world = base_xy.view(B,1,2) + torch.matmul(hip_offsets_body, R_yaw.transpose(1,2))  # (B,4,2)

        # v_des: cmd in body -> world
        if hasattr(self, "cmd_rand"):
            v_des_body = self.cmd_rand[:, 0:2]                   # (B,2)
        else:
            v_des_body = torch.stack([self.vx_star, torch.zeros_like(self.vx_star)], dim=-1)
        v_des_world = torch.einsum("bij,bj->bi", R_yaw, v_des_body)  # (B,2)

        # v_now: current estimated base velocity (world)
        v_now_world = self.base_lin_world[:, 0:2]                # (B,2)

        term_predict = v_now_world.view(B,1,2) * T_left.unsqueeze(-1)                 # v*(1-p)T_swing
        term_ff      = 0.5 * v_des_world.view(B,1,2) * T_stance.view(B,1,1)           # 0.5*T_stance*v_des
        #term_fb      = cfg.k_raibert * (v_now_world - v_des_world).view(B,1,2)        # +k*(v - v_des)
        term_fb      = cfg.k_raibert * (v_now_world - v_des_world).view(B,1,2)        # +k*(v - v_des)
        fb_clip = float(getattr(cfg, "raibert_fb_clip", 0.15))
        term_fb = term_fb.clamp(min=-fb_clip, max=+fb_clip)


        # x_bias along body forward projected to world
        fwd_world = torch.stack([cy, sy], dim=1)                 # (B,2)
        term_bias = cfg.x_bias * fwd_world.view(B,1,2)
        #p_land_xy_world = p_hip_xy_world + term_predict + term_ff + term_fb + term_bias
        p_land_xy_world = p_hip_xy_world  + term_ff 
        return p_hip_xy_world, p_land_xy_world


    #落脚点计算（机体系 Raibert 版本）
    def _raibert_footholds_body(self, T_stance):
        """
        纯机体系版本的 Raibert 落脚点（batch）
        返回:
            p_hip_xy_world : (B,4,2)
            p_land_xy_world: (B,4,2)
        """
        dev = self.device
        B = self.B

        # ===== 把 T_stance 统一成 (B,1) 方便广播 =====
        if not torch.is_tensor(T_stance):
            T = torch.full((B, 1), float(T_stance), device=dev)
        else:
            T = T_stance.to(dev)
            if T.dim() == 0:
                T = T.view(1, 1).expand(B, 1)
            elif T.dim() == 1:
                assert T.shape[0] == B, f"T_stance.shape[0]={T.shape[0]} != B={B}"
                T = T.view(B, 1)
            else:
                raise ValueError(f"Unexpected T_stance shape: {T.shape}")

        # yaw (B,)
        qw, qx, qy, qz = (self.base_quat[:, 0], self.base_quat[:, 1],
                          self.base_quat[:, 2], self.base_quat[:, 3])
        siny_cosp = 2.0 * (qw*qz + qx*qy)
        cosy_cosp = 1.0 - 2.0 * (qy*qy + qz*qz)
        yaw = torch.atan2(siny_cosp, cosy_cosp)

        cy = torch.cos(yaw)
        sy = torch.sin(yaw)
        R_yaw = torch.stack([
            torch.stack([cy, -sy], dim=-1),
            torch.stack([sy,  cy], dim=-1)
        ], dim=1)  # (B,2,2)

        hip_offsets_body = torch.tensor([
            [ +0.1934, +0.1420 ],
            [ +0.1934, -0.1420 ],
            [ -0.1934, +0.1420 ],
            [ -0.1934, -0.1420 ],
        ], dtype=torch.float32, device=dev).view(1,4,2).expand(B,4,2)

        base_xy = self.base_pos[:, 0:2]

        # ---- ✅ 命令速度：优先用 cmd_rand 的 vx,vy（机体系）----
        if hasattr(self, "cmd_rand"):
            v_cmd_body_xy = self.cmd_rand[:, 0:2]  # (B,2)
        else:
            vx_star = self.vx_star
            v_cmd_body_xy = torch.stack([vx_star, torch.zeros_like(vx_star)], dim=-1)
        v_body_xy = self.base_lin_body[:, 0:2]  # (B,2)

        if self.cfg.use_paper_raibert:
            # Raibert: p_land = p_hip + 0.5*T_stance*v_des
            delta_body = 0.5 * T * v_cmd_body_xy
        else:
            forward_B = 0.5 * T * v_cmd_body_xy
            v_err_xy  = v_body_xy - v_cmd_body_xy
            fbk_B     = -self.cfg.k_raibert * v_err_xy
            x_bias_B  = torch.stack([
                torch.full((B,), self.cfg.x_bias, device=dev),
                torch.zeros(B, device=dev)
            ], dim=-1)
            delta_body = forward_B + fbk_B + x_bias_B

        p_hip_xy_body  = hip_offsets_body
        p_land_xy_body = p_hip_xy_body + delta_body.unsqueeze(1)

        # body -> world
        p_hip_xy_world  = base_xy.unsqueeze(1) + torch.matmul(p_hip_xy_body,  R_yaw.transpose(1,2))
        p_land_xy_world = base_xy.unsqueeze(1) + torch.matmul(p_land_xy_body, R_yaw.transpose(1,2))

        return p_hip_xy_world, p_land_xy_world
    
    # ---------------- stance helpers ----------------
    def _phase_u(self, phases: torch.Tensor) -> torch.Tensor:
        """
        phases: (B,4) rad
        return u in [0,1): (B,4)
        """
        return torch.remainder(phases, 2 * math.pi) / (2 * math.pi)
    
    def _get_beta_minfeet_allow_aerial(self):
        """
        按图 9.2/9.3：每个 gait 给 duty factor β=r，并决定是否允许腾空段（min_feet=0）
        返回:
        beta_B      : (B,) in (0,1]
        min_feet_B  : (B,) int64, 0/2/4
        allow_aerial: (B,) bool
        """
        dev, B = self.device, self.B
        cfg = self.cfg

        # 你现有 gait_ids: 0 stand, 1 trot, 2 pace, 3 bound, 4 gallop  :contentReference[oaicite:1]{index=1}
        gait_ids = getattr(self, "gait_ids", torch.ones(B, dtype=torch.long, device=dev)).to(dev)

        # ---- 默认 duty factors（可按你教材/论文改）----
        beta_stand  = 1.0
        beta_trot_n = 0.5   # 图 9.2a
        beta_trot_w = 0.6   # 图 9.2b: r>0.5 (walking)
        beta_trot_r = 0.4   # 图 9.2c: r<0.5 (running -> 有腾空)
        beta_pace   = 0.5   # 图 9.3b 文中给 r=0.5
        beta_bound  = 0.4   # 图 9.3a 文中给 r<0.5 (有腾空)
        beta_gallop = 0.35  # 教材图未给具体值，常用更小 duty 让腾空明显
        # 你可以通过 cfg.trot_style 强制选哪一种 trot 变体： "normal" / "walk" / "run"
        trot_style = getattr(cfg, "trot_style", "normal")
        if trot_style not in ("normal", "walk", "run"):
            trot_style = "normal"

        beta_trot = {"normal": beta_trot_n, "walk": beta_trot_w, "run": beta_trot_r}[trot_style]

        # ---- 组装 beta 表（按 gait_id）----
        beta_table = torch.tensor(
            [beta_stand, beta_trot, beta_pace, beta_bound, beta_gallop],
            dtype=torch.float32, device=dev
        )
        beta_B = beta_table[gait_ids].clamp(min=1e-3, max=1.0)  # (B,)

        # ---- min_feet：严格按“是否允许腾空段”决定 ----
        # walking/normal trot、pace：任意时刻至少 2 足支撑
        # stand：4 足
        # bound/gallop、running trot：允许腾空 => min_feet=0
        allow_aerial = (gait_ids == 3) | (gait_ids == 4)  # bound/gallop
        # trot 的 aerial 由 beta<0.5 决定（running trot）
        allow_aerial = allow_aerial | ((gait_ids == 1) & (beta_B < 0.5))

        # 训练 warm-up：先禁用腾空段，避免开局“自由落体+前翻”
        if getattr(cfg, "train_no_aerial", False):
            allow_aerial = torch.zeros_like(allow_aerial, dtype=torch.bool)
            # ✅ 关键修复：禁腾空时，强制 duty >= 0.5
            # 否则 bound/gallop 的 β<0.5 会造成“名义 stance 少于 2 足”，
            # _mix_stance 会用 topk 硬补支撑脚 -> 假支撑脚锁 last_contact -> 易前翻
            beta_B = torch.clamp(beta_B, min=0.5, max=1.0)

        min_feet_B = torch.full((B,), 2, dtype=torch.long, device=dev)
        min_feet_B = torch.where(gait_ids == 0, torch.full_like(min_feet_B, 4), min_feet_B)
        min_feet_B = torch.where(allow_aerial, torch.zeros_like(min_feet_B), min_feet_B)
        return beta_B, min_feet_B, allow_aerial
    
    def _stance_phase_mask(self, phases: torch.Tensor, beta_B: torch.Tensor) -> torch.Tensor:
        """
        严格 duty-factor 定义：
          u in [0,1), stance if u < β, swing otherwise
        phases: (B,4)
        beta_B: (B,)
        return: (B,4,1)
        """
        B = self.B
        u = self._phase_u(phases)                       # (B,4)
        beta = beta_B.view(B, 1)                        # (B,1)
        return (u < beta).float().unsqueeze(-1)         # (B,4,1)
    

    def _mix_stance(self,
                phases: torch.Tensor,
                contact_flags: torch.Tensor,
                beta_B: torch.Tensor,
                min_feet_B: torch.Tensor,
                w_phase: float = 1.0,
                w_contact: float = 0.0):
        """
        phases       : (B,4)
        contact_flags: (B,4,1)  (可传进来，但 strict 模式下默认 w_contact=0)
        beta_B       : (B,)
        min_feet_B   : (B,) long, 0/2/4

        返回 stance_mask: (B,4,1)
        """
        dev, B = self.device, self.B

        phase_mask = self._stance_phase_mask(phases, beta_B)        # (B,4,1)

        # strict: w_contact=0, w_phase=1 => 纯按图定义
        mix = w_phase * phase_mask + w_contact * contact_flags
        mix = torch.clamp(mix, 0.0, 1.0)                            # (B,4,1)

        flat = mix.view(B, 4)
        # per-env 强制最少支撑足数（running/bound/gallop 时 min_feet_B=0，不会补足 => 可以腾空）
        for b in range(B):
            k = int(min_feet_B[b].item())
            if k <= 0:
                continue
            if float(flat[b].sum()) < k:
                topk = torch.topk(flat[b], k=k).indices
                out = torch.zeros_like(flat[b])
                out[topk] = 1.0
                flat[b] = out
        return flat.view(B, 4, 1)

    # ---------------- PD -> foot forces ----------------
    def estimate_foot_forces(self, q_ref12, q_now12, qd_now12, stance_mask):
        """
        q_ref12, q_now12, qd_now12 : (B,12)
        stance_mask: (B,4,1)
        返回 f: (B,4,3)
        """
        dev = self.device
        Kp, Kd = self.cfg.pd_kp, self.cfg.pd_kd

        tau = Kp * (q_ref12 - q_now12) - Kd * qd_now12  # (B,12)
        tau = tau.view(self.B, 12, 1)

        J_all = self.foot_jacobians()                   # (B,4,3,cols)
        dof_offset = getattr(self, "jac_dof_offset", 0)
        J12 = J_all[..., self.ctrl_idx_t + dof_offset]  # (B,4,3,12) 关节 DOF 对应的列

        f_list = []
        I = torch.diag(torch.tensor(
            [self.cfg.Ixx, self.cfg.Iyy, self.cfg.Izz], device=dev
        ))

        for b in range(self.B):
            Jb = J12[b]                          # (4,3,12)
            wmask = stance_mask[b].view(4,1,1)   # (4,1,1)
            Jw = Jb * wmask                      # (4,3,12)

            Jbig = torch.cat([Jw[i] for i in range(4)], dim=0)  # (12,12)
            JJt  = Jbig @ Jbig.T
            rhs  = Jbig @ tau[b]                                # (12,1)

            U, S, Vh = torch.linalg.svd(JJt + 1e-9*torch.eye(12, device=dev))
            S = torch.clamp(S, min=1e-3)
            Ainv = U @ torch.diag_embed(1.0/S) @ Vh
            y = Ainv @ rhs
            f_b = y.view(4,3)

            # 摩擦锥 + Fz 限幅
            fz = f_b[:, 2:3]
            fz = torch.clamp(fz, min=self.cfg.fz_min, max=self.cfg.fz_max) * stance_mask[b]
            ft = f_b[:, :2]
            ft_norm = torch.linalg.norm(ft, dim=-1, keepdim=True)
            ft_max = self.cfg.mu_tangent * fz
            scale = torch.clamp(ft_max / (ft_norm + 1e-6), max=1.0)
            ft = ft * scale
            f_b = torch.cat([ft, fz], dim=-1)   # (4,3)
            f_list.append(f_b)

        f = torch.stack(f_list, dim=0)  # (B,4,3)
        return f

    # ---------------- SRBD ----------------
    def _srbd_init_from_isaac(self):
        """Initialize full 3D SRBD state from Isaac (batch)."""
        dev = self.device
        self.srbd_p = self.base_pos.clone()            # (B,3)
        self.srbd_v = self.base_lin_world.clone()      # (B,3)
        self.srbd_q = self.base_quat.clone()           # (B,4)

        w_list = []
        for b in range(self.B):
            w_b = quat_rotate_inverse_wxyz(self.srbd_q[b],
                                           self.base_ang_world[b],
                                           dev)
            w_list.append(w_b)
        self.srbd_w = torch.stack(w_list, dim=0)       # (B,3)

    def _quat_norm(self, q):
        if q.dim() == 1:
            return q / (q.norm() + 1e-9)
        else:
            return q / (q.norm(dim=-1, keepdim=True) + 1e-9)

    def _srbd_step(self, f_world, q_ref12, dt):
        """
        Full 3D centroidal dynamics step (batch, 无 inplace 写法).
        f_world: (B,4,3)
        q_ref12: (B,12)
        """
        dev = self.device
        m, g = self.cfg.m, self.cfg.g

        # 当前 SRBD 状态的快照（不要在中途修改 self.srbd_*）
        p = self.srbd_p          # (B,3)
        v = self.srbd_v          # (B,3)
        q = self.srbd_q          # (B,4)
        w = self.srbd_w          # (B,3)

        # ---- 平动部分 ----
        Fsum = f_world.sum(dim=1) + torch.tensor(
            [0.0, 0.0, -m * g], device=dev
        ).view(1, 3)                               # (B,3)
        a = Fsum / m                                # (B,3)

        # 先用“旧状态”计算足端位置和力矩
        #p_foot = self.foot_positions_srbd(q_ref12.detach())  # (B,4,3)
        p_foot = self.foot_positions_srbd(q_ref12)

        I = torch.diag(torch.tensor(
            [self.cfg.Ixx, self.cfg.Iyy, self.cfg.Izz], device=dev
        ))                                          # (3,3)

        q_new_list = []
        w_new_list = []

        for b in range(self.B):
            # 臂长 r: from COM to foot
            r = p_foot[b] - p[b].view(1, 3)        # (4,3)
            tau_world = torch.cross(r, f_world[b], dim=-1).sum(dim=0)  # (3,)

            # world -> body
            qw, qx, qy, qz = q[b]
            R = quat_to_rot(qw, qx, qy, qz, dev)   # (3,3)
            tau_body = R.t() @ tau_world           # (3,)

            # 欧拉方程 I * wdot = tau - w × (I w)
            w_b = w[b]                             # (3,)
            
            Iw = I @ w_b                           # (3,)
            wdot = torch.linalg.solve(
                I, tau_body - torch.linalg.cross(w_b, Iw)
            )                                      # (3,)
            

            

            w_b_new = w_b + wdot * dt              # (3,)

            wx, wy, wz = w_b_new.unbind()
            z = torch.zeros_like(wx)
            Omega = torch.stack([
                torch.stack([z,  -wx, -wy, -wz]),
                torch.stack([wx,  z,   wz, -wy]),
                torch.stack([wy, -wz,  z,   wx]),
                torch.stack([wz,  wy, -wx,  z ]),
            ], dim=0)
            qdot = 0.5 * (Omega @ q[b])
            q_b_new = self._quat_norm(q[b] + qdot * dt)

            w_new_list.append(w_b_new)
            q_new_list.append(q_b_new)

        # ---- 真正更新 self.srbd_*，一次性整体赋值（无 inplace 切片） ----
        p_new = p + v * dt                         # (B,3)
        v_new = v + a * dt                         # (B,3)
        q_new = torch.stack(q_new_list, dim=0)     # (B,4)
        w_new = torch.stack(w_new_list, dim=0)     # (B,3)

        self.srbd_p = p_new
        self.srbd_v = v_new
        self.srbd_q = q_new
        self.srbd_w = w_new


    # ---------------- SRBD-based foot positions ----------------
    def foot_positions_srbd(self, q_ref12):
        """
        3D SRBD 足端位置 (batch)
        q_ref12: (B,12)
        返回 (B,4,3)
        """
        dev = self.device
        B = self.B
        p_base = self.srbd_p          # (B,3)
        q = self.srbd_q               # (B,4)

        hip_offsets = torch.tensor([
            [ +0.1934, +0.1420, 0.0 ],
            [ +0.1934, -0.1420, 0.0 ],
            [ -0.1934, +0.1420, 0.0 ],
            [ -0.1934, -0.1420, 0.0 ],
        ], device=dev)                # (4,3)

        #L1, L2 = 0.25, 0.25
        L1, L2 = 0.213, 0.213
        out = torch.zeros(B, 4, 3, device=dev)

        for b in range(B):
            qb = q[b]
            R = quat_to_rot(qb[0], qb[1], qb[2], qb[3], dev)

            hip_world = p_base[b].view(1,3) + (R @ hip_offsets.t()).t()  # (4,3)

            q_leg = q_ref12[b].view(4,3)
            q2, q3 = q_leg[:,1], q_leg[:,2]
            x = L1*torch.sin(q2) + L2*torch.sin(q2+q3)
            z = -L1*torch.cos(q2) - L2*torch.cos(q2+q3)
            y = torch.zeros_like(x)
            off_body = torch.stack([x,y,z], dim=1)  # (4,3)
            off_world = (R @ off_body.t()).t()
            out[b] = hip_world + off_world

        return out  # (B,4,3)

    # Raibert 名义参考关节角 (batch)
    def _raibert_qref(self):
        dev = self.device
        B = self.B
        cfg = self.cfg

        # -------- phases (B,4) --------
        if hasattr(self, "leg_phase_offsets_B"):
            phase_offsets = self.leg_phase_offsets_B  # (B,4)
        else:
            phase_offsets = self.leg_phase_offsets.view(1, 4).repeat(B, 1)
        phases = phase_offsets + self.phase.view(B, 1)  # (B,4)

        # 整周期扫掠（推进相关），以及仅 swing 的抬腿（清障相关）
        s_full  = torch.sin(phases)                      # (B,4) in [-1,1]
        s_swing = torch.clamp(s_full, min=0.0)           # (B,4) in [0,1]

        # -------- 速度 & deadzone --------
        if hasattr(self, "cmd_rand"):
            v_cmd_xy = self.cmd_rand[:, 0:2]             # (B,2)
            v_mag = torch.linalg.norm(v_cmd_xy, dim=1)   # (B,)
        else:
            v_mag = self.vx_star.abs()
        dead = float(getattr(cfg, "cmd_deadzone", 0.05))
        move_mask = (v_mag >= dead).float()              # (B,)

        v_hi = max(0.20, float(getattr(cfg, "vx_max", 0.5)))
        spd_s = torch.clamp(v_mag / (v_hi + 1e-6), 0.0, 1.0)  # (B,)
        amp_scale = (0.50 + 0.40 * spd_s) * move_mask         # (B,)
        #amp_scale = (0.80 + 0.50 * spd_s) * move_mask   # 低速也有足够摆幅
        #amp_scale = torch.clamp(amp_scale, 0.0, 1.10)
        
        
        # -------- gait_id 分支 --------
        gait_ids = getattr(self, "gait_ids",
                           torch.ones(B, dtype=torch.long, device=dev)).to(dev) 


        # 粗糙地形缩放
        rough = 1.0 if bool(globals().get("USE_COMPLEX_TERRAIN", False)) else 0.0
        sweep_scale  = (1.0 - 0.15 * rough)   # 少一点前后扫掠，降低打滑
        lift_scale   = (1.0 + 0.10 * rough)   # 多一点抬腿
        hipab_scale  = (1.0 + 0.20 * rough)   # 略增外展，提升清障/稳定

        #  每个 gait 的（扫掠/抬腿）幅度表（单位：rad）
        # stand=0；trot/pace 适中；bound/gallop 更大
        #th_sweep_tbl：每种 gait 的 thigh（大腿关节）“扫掠/摆动”幅度表（单位 rad）。
        #ca_sweep_tbl：每种 gait 的 calf（小腿关节）“扫掠/摆动”幅度表（单位 rad）。
        th_sweep_tbl = torch.tensor([0.00, 0.16, 0.12, 0.22, 0.24], device=dev) #0.18 pace目前最稳定的
        ca_sweep_tbl = torch.tensor([0.00, 0.45, 0.25, 0.38, 0.42], device=dev) #0.25 pace目前最稳定的
        

        # 抬腿幅度：整体减小一档
        #th_lift_tbl：每种 gait 的 thigh 在 swing 期额外抬腿的幅度（rad）
        #ca_lift_tbl：每种 gait 的 calf 在 swing 期额外抬腿/收腿的幅度（rad）
        th_lift_tbl  = torch.tensor([0.00, 0.00, 0.08, 0.11, 0.13], device=dev)  # 0.08
        ca_lift_tbl  = torch.tensor([0.00, 0.00, 0.18, 0.23, 0.26], device=dev)  # 0.18
        

        # hip 外展幅度（pace 稍大，避免擦腿）
        #hip_abd_tbl  = torch.tensor([0.00, 0.03, 0.05, 0.02, 0.02], device=dev)
        hip_abd_tbl  = torch.tensor([0.00, 0.0, 0.07, 0.02, 0.02], device=dev)
        

        th_sweep = th_sweep_tbl[gait_ids] * amp_scale * sweep_scale   # (B,)
        ca_sweep = ca_sweep_tbl[gait_ids] * amp_scale * sweep_scale   # (B,)
        th_lift  = th_lift_tbl[gait_ids]  * amp_scale * lift_scale    # (B,)
        ca_lift  = ca_lift_tbl[gait_ids]  * amp_scale * lift_scale    # (B,)

        hip_abd  = hip_abd_tbl[gait_ids]  * amp_scale * hipab_scale   # (B,)
        
        
        #---------------------------------------------------------------------
        # 新增的部分：
        # -------- bound / gallop 时，专门削弱后腿（RL, RR）的扫掠/抬腿幅度 --------
        # gait_ids: 0=stand, 1=trot, 2=pace, 3=bound, 4=gallop
        hind_gain_env = torch.ones(B, device=dev)
        is_bound_or_gallop = (gait_ids == 2) | (gait_ids == 3) |(gait_ids == 4) 
        hind_gain_env = torch.where(
            is_bound_or_gallop,
            torch.full_like(hind_gain_env, 0.65),   # 例如 0.65，可以之后再调
            hind_gain_env
        )  # (B,)

        # 展开成 (B,4)，只作用在 RL, RR（索引 2,3），前腿保持 1.0
        hind_gain_legs = torch.ones(B, 4, device=dev)
        hind_gain_legs[:, 2] = hind_gain_env   # RL
        hind_gain_legs[:, 3] = hind_gain_env   # RR

        # 对相位因子做 per-leg 缩放：等价于“后腿幅度更小”
        s_full_scaled  = s_full  * hind_gain_legs  # (B,4)
        s_swing_scaled = s_swing * hind_gain_legs  # (B,4)


        #---------------------------------------------------------------------
        

        # -------- 名义 q_ref_all: (B,4,3) -> (B,12) --------
        q_ref_single = torch.as_tensor(self.q_default_np, device=dev).clone()  # (12,)
        q_ref_all = q_ref_single.view(1, 4, 3).repeat(B, 1, 1)                 # (B,4,3)

        # 1) thigh/calf：整周期扫掠（推进） + swing 抬腿（清障）
        # thigh idx=1, calf idx=2
        """
        q_ref_all[:, :, 1] += th_sweep.view(B, 1) * s_full + th_lift.view(B, 1) * s_swing
        q_ref_all[:, :, 2] -= ca_sweep.view(B, 1) * s_full + ca_lift.view(B, 1) * s_swing
4) 如果 /mujoco/lowcmd 也在变，但 MuJoCo 里还是不动：就查 “执行” 侧
这时问题通常是这几类（按高概率排序）：

A. mujoco_simulator 没把 lowcmd 写进 data.ctrl（或写错索引）
在 mujoco_simulator 里加两条打印（只在变化时打印最好）：

打印你写进 MuJoCo 的 ctrl 前 3 个


        """
    

        q_ref_all[:, :, 1] += th_sweep.view(B, 1) * s_full_scaled  + th_lift.view(B, 1)  * s_swing_scaled
        q_ref_all[:, :, 2] -= ca_sweep.view(B, 1) * s_full_scaled  + ca_lift.view(B, 1)  * s_swing_scaled
        
        
        # 2) hip：swing 期做一点外展/内收（左右交替），增强清障与姿态稳定
        hip_sign = torch.tensor([+1.0, -1.0, +1.0, -1.0], device=dev).view(1, 4)  # FL,FR,RL,RR
        q_ref_all[:, :, 0] += hip_abd.view(B, 1) * hip_sign * s_swing

        # stand gait 强制回默认（可选但更稳）
        stand_mask = (gait_ids == 0).float().view(B, 1, 1)
        q_ref_all = q_ref_all * (1.0 - stand_mask) + q_ref_single.view(1, 4, 3) * stand_mask

        return q_ref_all.view(B, 12)
    # ---------------- env step ----------------
    def step(self, delta_q: torch.Tensor):
        """
        delta_q: (B,12)
        """
        cfg = self.cfg

        # 根据当前 vx_star 更新每只狗的步频 / 摆动高度
        self._update_gait_from_cmd()                      # ★ 新增

        # 使用 per-env 步频更新相位：phase_{t+1} = phase_t + 2π f Δt
        # self.step_freq_B: (B,)
        move = getattr(self, "move_mask_B", torch.ones(self.B, device=self.device))
        self.phase = (self.phase + 2*math.pi*self.step_freq_B*cfg.dt*move) % (2*math.pi)

        # Raibert 名义姿态
        q_nom   = self._raibert_qref()     # (B,12)
        #q_nom = torch.as_tensor(self.q_default_np, device=self.device).view(1, 12).repeat(self.B, 1)  # (B,12)
        #delta_q = torch.tanh(delta_q) * 0.20
        delta_q = torch.tanh(delta_q) * 0.10

        #---------------------------------------
        # stop env 判定
        dead = float(getattr(cfg, "cmd_deadzone", 0.05))
        v_mag = torch.linalg.norm(self.cmd_rand[:, 0:2], dim=1)
        stop_env = (v_mag < dead)  # (B,)
        # stop 时：不允许策略扰动站立
        delta_q = torch.where(stop_env[:, None], torch.zeros_like(delta_q), delta_q)
        #---------------------------------------

        q_ref12 = q_nom + delta_q          # (B,12)

        # 目标关节
        target_slice = self.local_targets.clone()   # (B,dof)
        target_slice[:, self.ctrl_idx_t] = q_ref12 * self.ctrl_sign   # (B,12)

        # 限幅 & 限位 & 变化率
        target_slice = torch.max(torch.min(target_slice, self._hi_slice), self._lo_slice)
        target_slice = self._limit_step(self.local_targets, target_slice, max_step=0.05)
        self.local_targets = target_slice
        self.pos_targets_batch[:] = self.local_targets
        self._commit_pos_targets()

        # Isaac 仿真一步
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        if self.viewer is not None:
            self._render_cnt += 1
            if self._render_cnt % self.render_every == 0:
                self.gym.step_graphics(self.sim) 
                self._update_chase_camera() 
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if self.gym.query_viewer_has_closed(self.viewer):
                    self.gym.destroy_viewer(self.viewer); self.viewer = None

        self._update_cache()

        # 更新每条腿最近接触地面的高度
        with torch.no_grad():
            p_foot = self.foot_positions()              # (B,4,3)
            c = self.contact_flags().squeeze(-1)        # (B,4)
            prev_c = getattr(self, "prev_contact_flags", None)
            if prev_c is None:
                self.prev_contact_flags = c.unsqueeze(-1).clone()
                prev_c = self.prev_contact_flags

            # ✅ 修复：touchdown 只要发生 contact edge 就更新
            # 原来的 stance_phase gate 会让“擦地/提前触地/地形凸起触地”被忽略，
            # 导致 last_contact_* 长期锁错高度/位置，stance 锁脚时易产生前翻力矩
            touchdown = ((prev_c.squeeze(-1) <= 0.5) & (c > 0.5))  # (B,4)
            self.last_contact_xy = torch.where(
                touchdown.unsqueeze(-1),
                p_foot[..., 0:2],
                self.last_contact_xy
            )

            # ===== stride debug print (env0, per-leg touchdown-to-touchdown) =====
            b0 = 0
            if touchdown[b0].any():
                # 基本量：真实速度、步频
                # 注意：你这里 cmd_rand 是 body frame
                v_body_x = float(self.base_lin_body[b0, 0].item())
                v_body_y = float(self.base_lin_body[b0, 1].item())
                f0 = float(self.step_freq_B[b0].item()) if hasattr(self, "step_freq_B") else float(self.cfg.step_freq)
                cmd0 = self.cmd_rand[b0].detach().cpu().numpy() if hasattr(self, "cmd_rand") else None

                # ===== yaw & world forward/left (env0) =====
                yaw0 = float(self.yaw[b0].item())  # env0 base yaw (rad), from _update_cache()
                cy0, sy0 = math.cos(yaw0), math.sin(yaw0)
                fwd_w = torch.tensor([cy0, sy0], device=self.device, dtype=torch.float32)      # (2,)
                left_w = torch.tensor([-sy0, cy0], device=self.device, dtype=torch.float32)    # (2,)

                # 可选：打印一下 yaw rate（body frame z）
                omega_z0 = float(self.base_ang_body[b0, 2].item())  # rad/s


                
                # 名义 stride（按当前 f0 与真实 v_body_x 估计）
                stride_est = (v_body_x / (f0 + 1e-9))  # m
                step_est   = 0.5 * stride_est          # m

                leg_names = ["FL", "FR", "RL", "RR"]
                # ===== phase u (env0) for touchdown gating =====
                if hasattr(self, "leg_phase_offsets_B"):
                    phase_offsets = self.leg_phase_offsets_B
                else:
                    phase_offsets = self.leg_phase_offsets.view(1, 4).repeat(self.B, 1)
                
                phases = phase_offsets + self.phase.view(self.B, 1)    # (B,4)
                u0 = self._phase_u(phases)[b0]                         # (4,) in [0,1)
                u_eps = 0.08
                
                for leg in range(4):
                    if bool(touchdown[b0, leg].item()):
                        # ===== 相位门控：只承认“相位边界附近”的 touchdown（过滤抖动/二次触地）=====
                        # 真正 touchdown（swing->stance）应该发生在 u 接近 0 的时候
                        if not (float(u0[leg].item()) < u_eps):
                            continue
                        min_dt = 0.12  # 秒：保守的“同腿两次落地最小间隔”
                        min_steps = int(min_dt / self.cfg.dt)  # dt=0.002 -> 60步

                        # 或者更自适应（推荐）：半周期的 60% 作为冷却
                        T0 = 1.0 / max(f0, 1e-6)
                        min_steps = int( 0.5 * T0 / self.cfg.dt)  # 0.6*(T/2)

                        # 冷却判定
                        if (self.t - int(self._stride_last_td_step0[leg].item())) < min_steps:
                             continue  # 太近了，认为是抖动触发，忽略
                        self._stride_last_td_step0[leg] = self.t


                        xy = p_foot[b0, leg, 0:2].detach()  # (2,)
                        if self._stride_have_td0[leg]:
                            dxy = (xy - self._stride_last_td_xy0[leg])                    # (2,)
                            stride_xy  = torch.linalg.norm(dxy).item()                    # 欧氏距离
                            stride_fwd = torch.dot(dxy, fwd_w).item()                     # 前向投影（最关键）
                            stride_lat = torch.dot(dxy, left_w).item()                    # 侧向漂移（判断侧漂）

                            self._stride_count0[leg] += 1

                            if stride_fwd > 0.45 or stride_xy > 0.60:
                                # 异常：不打印，不计数，但更新缓存到当前点，防止后续连环爆炸
                                self._stride_last_td_xy0[leg] = xy
                                continue
                            # 控制刷屏：超过限制就不再打印（可关）
                            if (self._stride_print_limit is None) or (int(self._stride_count0[leg].item()) <= int(self._stride_print_limit)):
                                print(
                                    f"[STRIDE env0 t={self.t:05d}] {leg_names[leg]} "
                                    f"stride_xy={stride_xy:.3f}  fwd={stride_fwd:.3f}  lat={stride_lat:.3f} | "
                                    f"v_body=({v_body_x:+.2f},{v_body_y:+.2f}) f={f0:.3f} v/f≈{stride_est:.3f} | "
                                    f"yaw={yaw0:+.2f} wz={omega_z0:+.2f} | "
                                    f"cmd={np.round(cmd0,3) if cmd0 is not None else None}"
                                )
                        else:
                            # 第一次 touchdown：只记录不打印
                            self._stride_have_td0[leg] = True

                        # 更新该腿上次 touchdown 点
                        self._stride_last_td_xy0[leg] = xy

            self.last_contact_z = torch.where(touchdown, p_foot[..., 2], self.last_contact_z)

            # 更新 prev_contact
            self.prev_contact_flags = c.unsqueeze(-1).clone()
        self.t += 1

        # ===== 开局前翻定位打印（只看 env0）=====
        if DBG_INIT_FALL and (self.t <= DBG_INIT_FALL_STEPS) and (self.t % DBG_INIT_FALL_EVERY == 0):
             b = int(DBG_INIT_FALL_ENV)
             try:
                 beta_B, min_feet_B, allow_aerial_B = self._get_beta_minfeet_allow_aerial()
                 beta0 = float(beta_B[b].item())
                 k0 = int(min_feet_B[b].item())
                 allow0 = bool(allow_aerial_B[b].item()) if torch.is_tensor(allow_aerial_B) else bool(allow_aerial_B)
 
                 # phases -> 原始 phase stance（未 topk） vs mix stance（含 topk 补足）
                 if hasattr(self, "leg_phase_offsets_B"):
                     phase_offsets = self.leg_phase_offsets_B
                 else:
                     phase_offsets = self.leg_phase_offsets.view(1, 4).repeat(self.B, 1)
                 phases = phase_offsets + self.phase.view(self.B, 1)   # (B,4)
                 u = self._phase_u(phases)                              # (B,4)
                 raw_phase_stance = (u[b] < beta_B[b].view(1)).float()  # (4,)
 
                 stance_mix = self._mix_stance(
                     phases=phases[b:b+1],
                     contact_flags=self.contact_flags().detach()[b:b+1],
                     beta_B=beta_B[b:b+1],
                     min_feet_B=min_feet_B[b:b+1],
                     w_phase=1.0,
                     w_contact=0.0,
                 ).squeeze(0).squeeze(-1)                               # (4,)
 
                 forced = (raw_phase_stance.sum() < float(k0))
 
                 # 基本状态
                 cmd0 = self.cmd_rand[b].detach().cpu().numpy() if hasattr(self, "cmd_rand") else None
                 sf0 = float(getattr(self, "step_freq_B", torch.tensor([self.cfg.step_freq], device=self.device))[b].item())
                 sh0 = float(getattr(self, "swing_height_B", torch.tensor([self.cfg.swing_height], device=self.device))[b].item())
                 c0 = self.contact_flags().detach()[b].squeeze(-1).cpu().numpy()
                 z_foot0 = p_foot[b, :, 2].detach().cpu().numpy()
                 z_lc0 = self.last_contact_z[b].detach().cpu().numpy()
 
                 print(
                     f"[DBG_INIT_FALL t={self.t:04d}] "
                     f"gait={int(self.gait_ids[b].item()) if hasattr(self,'gait_ids') else -1} "
                     f"beta={beta0:.2f} minFeet={k0} allowAerial={allow0} forcedTopK={bool(forced)} | "
                     f"cmd(vx,vy,yaw)={np.round(cmd0,3) if cmd0 is not None else None} "
                     f"f={sf0:.2f}Hz hSwing={sh0:.3f} | "
                     f"z={float(self.base_pos[b,2].item()):.3f} "
                     f"roll={float(self.roll[b].item()):+.3f} pitch={float(self.pitch[b].item()):+.3f} | "
                     f"contact={np.round(c0,0).astype(int).tolist()} "
                     f"stanceRaw={np.round(raw_phase_stance.cpu().numpy(),0).astype(int).tolist()} "
                     f"stanceMix={np.round(stance_mix.cpu().numpy(),0).astype(int).tolist()} | "
                     f"zFoot={np.round(z_foot0,3).tolist()} zLast={np.round(z_lc0,3).tolist()}"
                 )
             except Exception as _e:
                 # 不让 debug 打印影响训练
                 pass

        fallen_height = (self.base_pos[:, 2] < 0.16)           # (B,)
        fallen_tilt   = (torch.abs(self.roll) > 0.9) | (torch.abs(self.pitch) > 0.9)
        done = fallen_height | fallen_tilt                     # (B,)

        obs = self.get_obs()
        extra = {
            "done": done,
            "muN": torch.ones(self.B, device=self.device),
            "q_err_norm": torch.zeros(self.B, device=self.device),
        }
        q_now12 = self.q[:, self.ctrl_idx_t]                   # (B,12)
        q_err = q_ref12 - q_now12
        return obs, extra, q_err, q_ref12

    @torch.no_grad()
    def get_obs(self):
        cfg, dev = self.cfg, self.device
        B = self.B

        # cmd: [vx_cmd, vy_cmd, yaw_rate_cmd] 直接来自 reset 时采样的高层指令
        if hasattr(self, "cmd_rand"):
            cmd = self.cmd_rand.clone()                       # (B,3)
        else:
            # 保险 fallback：只有 vx_star 时的兼容写法
            cmd = torch.stack([
                self.vx_star,
                torch.zeros_like(self.vx_star),
                torch.zeros_like(self.vx_star)
            ], dim=1)

        #phases = self.leg_phase_offsets.view(1,4) + self.phase.view(B,1)  # (B,4)
        # 使用多步态相位表生成每条腿的相位
        if hasattr(self, "leg_phase_offsets_B"):
            phase_offsets = self.leg_phase_offsets_B                         # (B,4)
        else:
            phase_offsets = self.leg_phase_offsets.view(1,4).repeat(B,1)     # (B,4)

        phases = phase_offsets + self.phase.view(B,1)  # (B,4)




        sincos = torch.stack([torch.sin(phases), torch.cos(phases)], dim=2).reshape(B, 8)

        v_b = self.base_lin_body                             # (B,3)
        q_wxyz = self.base_quat                              # (B,4)
        w_b  = self.base_ang_body                            # (B,3)

        # 重力投影 (B,3)
        g_list = []
        for b in range(B):
            g_list.append(project_gravity_to_body(q_wxyz[b], cfg.g, dev))
        g_proj = torch.stack(g_list, dim=0)

        q_now12 = self.q[:, self.ctrl_idx_t].to(dev)         # (B,12)
        q_default = torch.from_numpy(self.q_default_np).to(dev).view(1,12)
        q_delta = (q_now12 - q_default * self.ctrl_sign.view(1,12))      # (B,12)

        # 总维度：3(cmd) + 8(phase) + 3(v_b) + 4(q) + 3(w_b) + 12(q_delta) + 3(g_proj) = 36
        obs = torch.cat([cmd, sincos, v_b, q_wxyz, w_b, q_delta, g_proj], dim=-1)  # (B,36)
        return obs

# ---------------- Training ----------------
def train(num_iters=1000, steps_per_iter=24,
          device="cuda" if torch.cuda.is_available() else "cpu",
          seed: int = None, smooth_k: int = 25):

    if seed is None: seed = int(os.getenv("SEED", 0))
    set_seed(seed)

    cfg = EnvCfg()
    cfg.trot_style = "walk"   # or "normal" / "run"
    cfg.rand_cmd = True          # ✅ 开启随机速度指令
    cfg.vx_min = +0.5         # 你可以改成论文的命令范围
    cfg.vx_max = +1.0
    cfg.train_no_aerial = True


    if PURE_PAPER_MODE:
        cfg.use_paper_raibert = True 

    env = RealQuadEnv(cfg, device=device)
    B = env.B
    model = Policy(dim_obs=36, dim_action=12).to(device)
    opt = AdamW(model.parameters(), lr=1e-3)

    # 主损失 Eq.(5) 各项权重
    a1, a2, a3, a4, a5, a6 = 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
    
    
    # 摆/站/参考脚步加权（reward shaping，不进主 loss）
    w_swing, w_stance, w_footref = 1.0, 0.02, 0.4
    w_slip = 0.05
    w_swingvel = 0.05

    pbar = tqdm(range(num_iters), ncols=92)
    losses = []; rewards = []
    vx_iter_track = []

    loss_v_hist_iter = []
    loss_h_hist_iter = []
    loss_omega_hist_iter = []
    loss_ctrl_hist_iter = []
    loss_gproj_hist_iter = []
    loss_foot_hist_iter = []

    for it in pbar:
        # α 对齐调度
        if PURE_PAPER_MODE:
            env.cfg.alpha_align = 0.9
        else:
            alpha = 0.4 + 0.4 * (it / max(1, num_iters - 1))
            alpha = float(min(alpha, 0.8))
            env.cfg.alpha_align = alpha

        # 只在 iter=0 做一次全局 reset；之后依靠局部 reset_envs
        if it == 0:
            env.reset(it=it)
            model.reset()
        else:
            if not ONLY_ITERATE_NO_RESET:
                env.reset(it=it)
                model.reset()

        episodic_reward = 0.0
        stuck_counter = 0

        v_world_hist = []
        q_body_hist  = []
        pz_hist  = []
        ureg_hist = []
        omega_hist, gproj_hist = [], []
        foot_ref_hist = []
        tilt_hist = []
        cmd_hist = []       # ★ 记录每步的高层速度命令 [vx_cmd, vy_cmd, yaw_cmd]
        

        hx = None
        hx_hold = None

        a_prev = torch.zeros(B, 12, device=device)

        for t in range(steps_per_iter):
            # 观测 (B,36)
            s = env.get_obs().to(device)

            # RNN / action_hold 逻辑（原样保留）
            if PURE_PAPER_MODE:
                if (t % cfg.action_hold) == 0:
                    a, hx = model(s, hx)   # (B,12)
                    a_prev = a
                    hx_hold = hx.detach() if hx is not None else None
                else:
                    a = a_prev.detach()
                    hx = hx_hold
            else:
                if (t % cfg.action_hold) == 0:
                    a_raw, hx = model(s, hx)
                    a_smooth = 0.7 * a_prev + 0.3 * a_raw
                    a_prev = a_smooth.detach()
                    hx_hold = hx.detach() if hx is not None else None
                    a = a_smooth
                else:
                    a = a_prev
                    hx = hx_hold

            # IsaacGym 仿真一步
            obs, extra, q_err, qref = env.step(a)

            # stuck 判定（只看第 0 只狗）
            v_body_now0 = env.base_lin_body[0]
            if (v_body_now0.norm() < 0.02) and (env.base_pos[0,2] < 0.20):
                stuck_counter += 1
            else:
                stuck_counter = 0

            # 足端运动学
            J = env.foot_jacobians()  # (B,4,3,dof)
            dof_offset = getattr(env, "jac_dof_offset", 0)
            J_ctrl = J[..., env.ctrl_idx_t + dof_offset]      # (B,4,3,12)
            qd12 = env.qd[:, env.ctrl_idx_t]                  # (B,12)
            v_foot = torch.einsum('bfkj,bj->bfk', J_ctrl, qd12)  # (B,4,3)

            p_foot = env.foot_positions()                     # (B,4,3)

            # 相位 (B,4)
            if hasattr(env, "leg_phase_offsets_B"):
                phase_offsets = env.leg_phase_offsets_B
            else:
                phase_offsets = env.leg_phase_offsets.view(1,4).repeat(B,1)
            phases = phase_offsets + env.phase.view(B,1)      # (B,4)
            # ✅ 用新的“stance+swing+Raibert”函数，得到足端 target 和 stance_mask
            pref, stance_mask, vref_foot = env._update_foot_targets_from_command(
                phases, p_foot, return_vref=True
            )
            swing_mask  = 1.0 - stance_mask                   # (B,4,1)

            # ===== 摆/站损失 =====
            # 1) 摆腿“要抬到 swing_height 以上”
            clearance = p_foot[:,:,2:3] - env.last_contact_z.unsqueeze(-1)  # (B,4,1)
            if hasattr(env, "swing_height_B"):
                h_tar = env.swing_height_B.view(B, 1, 1)                     # (B,1,1)
            else:
                h_tar = torch.full((B,1,1), env.cfg.swing_height, device=env.device)


            loss_swing_step = (
                F.relu(env.cfg.swing_height - clearance)**2 * swing_mask
            ).mean(dim=(1,2))

            # 2) 摆腿时尽量贴合参考 target (pref)
            loss_ref_step = (
                ((p_foot - pref)**2).sum(dim=-1, keepdim=True) * swing_mask
            ).mean(dim=(1,2))

            # 3) 站立脚的“脚速越小越好”（防止打滑）
            vt = torch.linalg.norm(v_foot[..., :2], dim=-1, keepdim=True)
            loss_stance_step = (vt * stance_mask).mean(dim=(1,2))
            loss_slip = w_slip * loss_stance_step

            # 4) ✅ 新增：摆动脚速度贴合“教材摆线速度”(式 9.15)
            # v_foot: (B,4,3), vref_foot: (B,4,3)
            loss_swing_vel_step = (
                ((v_foot - vref_foot) ** 2).sum(dim=-1, keepdim=True) * swing_mask
            ).mean(dim=(1,2))


            # SRBD 步进 + α 对齐
            q12      = env.q[:, env.ctrl_idx_t]          # (B,12)
            qd12_now = env.qd[:, env.ctrl_idx_t]
            f_est = env.estimate_foot_forces(q_ref12=qref,
                                             q_now12=q12.detach(),
                                             qd_now12=qd12_now.detach(),
                                             stance_mask=stance_mask.detach())
            env._srbd_step(f_world=f_est, q_ref12=qref, dt=env.cfg.dt)

            alpha = env.cfg.alpha_align
            if env.cfg.use_strict_alpha_align:
                env.srbd_p = env.base_pos + alpha * (env.srbd_p - env.srbd_p.detach())
                env.srbd_v = env.base_lin + alpha * (env.srbd_v - env.srbd_v.detach())
                env.srbd_q = env.base_quat + alpha * (env.srbd_q - env.srbd_q.detach())
                env.srbd_q = env._quat_norm(env.srbd_q)
                env.srbd_w = env.base_ang_body + alpha * (env.srbd_w - env.srbd_w.detach())
            else:
                env.srbd_v = env.base_lin + alpha * (env.srbd_v - env.srbd_v.detach())
                env.srbd_p = env.srbd_p.clone()
                env.srbd_p[:,2] = env.base_pos[:,2] + alpha * (env.srbd_p[:,2] - env.srbd_p[:,2].detach())

            v_hat3 = env.srbd_v.clone()           # (B,3)
            pz_hat = env.srbd_p[:, 2]            # (B,)

            v_world_hist.append(v_hat3.clone())
            q_body_hist.append(env.srbd_q.clone())
            pz_hist.append(pz_hat.clone())
            cmd_hist.append(env.cmd_rand.clone())      # ★ 记录此时的 [vx_cmd, vy_cmd, yaw_cmd]
            ureg_hist.append(a.clone())
            

            #p_foot_srbd = env.foot_positions_srbd(qref)  # (B,4,3)
            p_foot_srbd = env.foot_positions_srbd(qref)  # (B,4,3)
            foot_err_vec = (p_foot_srbd - pref) * swing_mask
            foot_ref_hist.append(foot_err_vec.clone())

            # 角速度（body frame）
            w_body_list = []
            for b in range(B):
                w_b = quat_rotate_inverse_wxyz(env.srbd_q[b], env.srbd_w[b], env.device)
                w_body_list.append(w_b)
            #w_body_srbd = torch.stack(w_body_list, dim=0)  # (B,3)
            #omega_hist.append(w_body_srbd.clone())
            w_body_srbd = env.srbd_w.clone()  # 已经是 body frame
            omega_hist.append(w_body_srbd)

            # 重力投影（body frame）
            gproj_list = []
            for b in range(B):
                gproj_list.append(project_gravity_to_body(env.srbd_q[b], env.cfg.g, env.device))
            gproj_hist.append(torch.stack(gproj_list, dim=0))

            tilt_hist.append(0.7*torch.abs(env.pitch) + 0.3*torch.abs(env.roll))  # (B,)

            # reward（per-env，再取平均）
            v_body_dbg_list = []
            for b in range(B):
                R_bw_dbg = quat_to_rot(env.srbd_q[b][0], env.srbd_q[b][1],
                                       env.srbd_q[b][2], env.srbd_q[b][3], env.device)
                v_body_dbg_list.append(R_bw_dbg.t().matmul(v_hat3[b]))
            v_body_dbg = torch.stack(v_body_dbg_list, dim=0)  # (B,3)

            r_v = -(env.vx_star - v_body_dbg[:,0]).abs()      # (B,)
            r_u = -0.01 * a.detach().pow(2).mean(dim=1)       # (B,)
            r_stab = -0.8 * ((pz_hat - cfg.h0).abs() + tilt_hist[-1])

            r_t = (r_v + r_u + r_stab
                   - w_swing*loss_swing_step
                   - w_stance*loss_stance_step
                   - w_footref*loss_ref_step
                   - w_swingvel*loss_swing_vel_step)
            episodic_reward += float(r_t.mean().item())

            done = extra["done"]              # (B,)

            # ★ 在这里局部 reset 那几条狗；同时给这些 env 换新 vx_star + gait
            if done.any():
                fallen_ids = torch.nonzero(done, as_tuple=False).squeeze(-1)
                episodic_reward -= cfg.term_penalty * float(done.float().mean().item())
                env.reset_envs(fallen_ids)

        # ====== Eq.(5) 逐项损失 ======
        if v_world_hist:
            v_world_seq = torch.stack(v_world_hist)   # (T,B,3)
            q_seq = torch.stack(q_body_hist)         # (T,B,4)
            # q_seq: (T,B,4) in wxyz
            qw = q_seq[..., 0]
            qx = q_seq[..., 1]
            qy = q_seq[..., 2]
            qz = q_seq[..., 3]

            # yaw from quaternion (wxyz)
            siny_cosp = 2.0 * (qw*qz + qx*qy)
            cosy_cosp = 1.0 - 2.0 * (qy*qy + qz*qz)
            yaw_seq = torch.atan2(siny_cosp, cosy_cosp)   # (T,B)

            # reference yaw: the yaw at reset (no grad)
            yaw_ref = env.last_reset_yaw.detach().view(1, B).expand_as(yaw_seq)

            # wrap to [-pi, pi]
            yaw_err = torch.atan2(torch.sin(yaw_seq - yaw_ref), torch.cos(yaw_seq - yaw_ref))
            loss_yaw = (yaw_err ** 2).mean()


            T_steps = v_world_seq.shape[0]
            v_body_seq = torch.zeros_like(v_world_seq)

            # 世界 -> 机体系线速度
            for t_idx in range(T_steps):
                for b in range(B):
                    qw,qx,qy,qz = q_seq[t_idx, b]
                    R_bw = quat_to_rot(qw,qx,qy,qz, device)
                    v_body_seq[t_idx, b] = R_bw.t().matmul(v_world_seq[t_idx, b])

            # ★ v_ref：使用历史的 [vx_cmd, vy_cmd]
            vref_body = torch.zeros_like(v_body_seq)
            if cmd_hist:
                cmd_seq = torch.stack(cmd_hist)          # (T,B,3)
                vref_body[..., 0:2] = cmd_seq[..., 0:2]  # 跟踪 vx、vy
            else:
                # fallback：只用当前 vx_star
                vref_body[..., 0] = env.vx_star.view(1,B).expand(T_steps,B)

            # ★ 速度 tracking loss: vx & vy
            #loss_v = ((v_body_seq - vref_body) ** 2).sum(dim=-1).mean()
            loss_v = ((v_body_seq[..., :2] - vref_body[..., :2]) ** 2).sum(-1).mean()
            
            # ==== 新增：每只狗自己的平均 vx_body ====
            # v_body_seq: (T,B,3) -> 先在时间上平均 -> (B,3)
            vx_env = v_body_seq[..., 0].mean(dim=0)         # (B,)
            vx_env_np = vx_env.detach().cpu().numpy()       # numpy 方便打印

            # 原来的整体平均 vx（所有 env + 所有时间步）
            vx_for_plot = float(vx_env.mean().item())
            # 可选：打印每个 env 的 vx
            print(f"[Iter {it}] vx_body per env:", np.round(vx_env_np, 3))
        else:
            loss_v = torch.tensor(0.0, device=device); vx_for_plot = 0.0



        if pz_hist:
            pz_seq = torch.stack(pz_hist)  # (T,B)
            loss_h = (pz_seq - cfg.h0).abs().mean()
        else:
            loss_h = torch.tensor(0.0, device=device)

        if omega_hist:
            omega_seq = torch.stack(omega_hist)  # (T,B,3)

            # roll/pitch 正则项：希望横滚 / 俯仰角速度不要太大
            rollpitch_sq = (omega_seq[..., :2] ** 2).sum(dim=-1)   # (T,B)

            # ★ yaw 角速度跟踪：omega_z vs yaw_cmd
            if cmd_hist:
                cmd_seq = torch.stack(cmd_hist)          # (T,B,3)
                yaw_cmd_seq = cmd_seq[..., 2]            # (T,B)
            else:
                yaw_cmd_seq = torch.zeros_like(omega_seq[..., 2])

            yaw_err_sq = (omega_seq[..., 2] - yaw_cmd_seq) ** 2    # (T,B)

            # 合并：既跟踪 yaw，又惩罚 roll/pitch
            loss_omega = (rollpitch_sq + yaw_err_sq).mean()
        else:
            loss_omega = torch.tensor(0.0, device=device)


        if ureg_hist:
            u_seq = torch.stack(ureg_hist)  # (T,B,12)
            loss_ctrl = (u_seq ** 2).sum(dim=-1).mean()
        else:
            loss_ctrl = torch.tensor(0.0, device=device)

        if gproj_hist:
            gproj_seq = torch.stack(gproj_hist)  # (T,B,3)
            #g_xy = gproj_seq[..., :2]
            #loss_gproj = (g_xy ** 2).sum(dim=-1).mean()
            g_xy = gproj_seq[..., :2]
            # 归一化：把单位从 m/s^2 变成无量纲，避免这项天然比其他项大一个数量级
            g_xy_norm = g_xy / cfg.g   # cfg.g 一般是 9.81
            loss_gproj = (g_xy_norm ** 2).sum(dim=-1).mean()
        else:
            loss_gproj = torch.tensor(0.0, device=device)

        if foot_ref_hist:
            foot_err_seq = torch.stack(foot_ref_hist)  # (T,B,4,3)
            loss_foot = (foot_err_seq ** 2).sum(dim=-1).mean()
        else:
            loss_foot = torch.tensor(0.0, device=device)

        yaw_w = 0.1
        loss = (a1*loss_v +
                a2*loss_h +
                a3*loss_omega +
                a4*loss_ctrl +
                a5*loss_gproj +
                a6*loss_foot  +
                yaw_w * loss_yaw
                )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        # detach SRBD 状态
        env.srbd_p = env.srbd_p.detach()
        env.srbd_v = env.srbd_v.detach()
        env.srbd_q = env.srbd_q.detach()
        env.srbd_w = env.srbd_w.detach()

        loss_v_hist_iter.append(float(loss_v.detach().cpu()))
        loss_h_hist_iter.append(float(loss_h.detach().cpu()))
        loss_omega_hist_iter.append(float(loss_omega.detach().cpu()))
        loss_ctrl_hist_iter.append(float(loss_ctrl.detach().cpu()))
        loss_gproj_hist_iter.append(float(loss_gproj.detach().cpu()))
        loss_foot_hist_iter.append(float(loss_foot.detach().cpu()))

        vx_iter_track.append(vx_for_plot)
        losses.append(loss.item()); rewards.append(episodic_reward)
        pbar.set_description(f"Iter {it:4d} | loss {loss.item():.3f} | v_body_x {vx_for_plot:+.2f}")


    # ===== 保存曲线 =====
    V = np.array(vx_iter_track, dtype=np.float32)
    S = steps_per_iter

    np.save("vx_curve_srbd_align.npy", V)
    plt.figure(); plt.plot(V)
    plt.xlabel("Training Iteration"); plt.ylabel(f"Avg body vx over {S} steps (m/s)")
    plt.tight_layout(); plt.savefig("vx_curve_srbd_align.png")

    plt.figure(); plt.plot(moving_average(V, smooth_k))
    plt.xlabel("Training Iteration"); plt.ylabel("Avg body vx (moving avg)")
    plt.tight_layout(); plt.savefig("vx_curve_srbd_align_smooth.png")

    plt.figure(); plt.plot(losses); plt.xlabel("Iteration"); plt.ylabel("Loss")
    plt.tight_layout(); plt.savefig("loss_curve_srbd_align.png")

    loss_parts = {
        "loss_v":     np.array(loss_v_hist_iter,     dtype=np.float32),
        "loss_h":     np.array(loss_h_hist_iter,     dtype=np.float32),
        "loss_omega": np.array(loss_omega_hist_iter, dtype=np.float32),
        "loss_ctrl":  np.array(loss_ctrl_hist_iter,  dtype=np.float32),
        "loss_gproj": np.array(loss_gproj_hist_iter, dtype=np.float32),
        "loss_foot":  np.array(loss_foot_hist_iter,  dtype=np.float32),
    }

    for name, arr in loss_parts.items():
        np.save(f"{name}_srbd_align.npy", arr)
        plt.figure()
        plt.plot(arr)
        plt.xlabel("Iteration")
        plt.ylabel(name)
        plt.tight_layout()
        plt.savefig(f"{name}_curve_srbd_align.png")

    plt.figure()
    for name, arr in loss_parts.items():
        plt.plot(arr, label=name)
    plt.xlabel("Iteration")
    plt.ylabel("Loss components")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_components_curve_srbd_align.png")

    R = np.array(rewards, dtype=np.float32); np.save("rewards_srbd_align.npy", R)
    plt.figure(); plt.plot(moving_average(R, smooth_k))
    plt.xlabel("Training Iteration"); plt.ylabel("Reward (moving avg)")
    plt.tight_layout(); plt.savefig("reward_curve_srbd_align.png")

    torch.save(model.state_dict(), "quad_diffsim_srbd_align_multi_robot.pth")
    # ===== 额外导出 TorchScript（给 ROS2 部署用）=====
    model.eval()

    class PolicyActOnly(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            a, _ = self.m(x)
            return a
    
    wrapper = PolicyActOnly(model).to(device)
    
    example_obs = torch.zeros(1, 36, device=device)  # 你的 DiffSim obs_dim=36
    traced = torch.jit.trace(wrapper, example_obs)
    traced.save("quad_diffsim_srbd_align_multi_robot.pt")
    print("✅ Saved TorchScript: quad_diffsim_srbd_align_multi_robot.pt")
    # ===== 额外导出 TorchScript（给 ROS2 部署用）=====

    print("✅ Training done (MULTI robot SRBD + α-align, Eq.(5) loss, body-frame vx tracking).")
    
if __name__ == "__main__":
    train(num_iters=1000, steps_per_iter=24, seed=0)