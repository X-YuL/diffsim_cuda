import math
import random

import numpy as np

# ⚠️ CRITICAL: Isaac Gym must be imported BEFORE torch
try:
    from isaacgym import gymapi
except Exception:
    gymapi = None

import torch


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def moving_average(x: np.ndarray, k: int = 25) -> np.ndarray:
    if k <= 1:
        return x
    w = np.ones(k) / k
    return np.convolve(x, w, mode="same")


# Quaternion wxyz -> rotation matrix (non-batch version)
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


# Project gravity acceleration to body frame -> used for observation and gravity penalty in loss
def project_gravity_to_body(q_wxyz, g, device):
    # q_wxyz = (w,x,y,z)
    qw, qx, qy, qz = q_wxyz
    R = quat_to_rot(qw, qx, qy, qz, device)
    g_w = torch.tensor([0.0, 0.0, -g], dtype=torch.float32, device=device)
    return R.t().matmul(g_w)  # (3,)


# Euler angles -> quaternion
def quat_from_rpy(roll: float, pitch: float, yaw: float):
    # gymapi.Quat(x, y, z, w)
    cr, sr = math.cos(roll*0.5),  math.sin(roll*0.5)
    cp, sp = math.cos(pitch*0.5), math.sin(pitch*0.5)
    cy, sy = math.cos(yaw*0.5),   math.sin(yaw*0.5)
    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    return gymapi.Quat(x, y, z, w)


# World frame -> body frame rotation (single q, v)
def quat_rotate_inverse_wxyz(q_wxyz, v, device):
    """
    Rotate vector v from world frame to body frame: v_body = R(q)^T v_world
    q_wxyz: (4,), v: (3,)
    """
    qw, qx, qy, qz = q_wxyz
    R = quat_to_rot(qw, qx, qy, qz, device)  # world <- body
    return (R.t() @ v.view(3,)).view(3,)
