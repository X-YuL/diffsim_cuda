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
    """
    Convert quaternion components to rotation matrix.
    Supports both single scalars and batched tensors of shape (B,).
    """
    r00 = 1 - 2 * (qy*qy + qz*qz)
    r01 = 2 * (qx*qy - qz*qw)
    r02 = 2 * (qx*qz + qy*qw)
    r10 = 2 * (qx*qy + qz*qw)
    r11 = 1 - 2 * (qx*qx + qz*qz)
    r12 = 2 * (qy*qz - qx*qw)
    r20 = 2 * (qx*qz - qy*qw)
    r21 = 2 * (qy*qz + qx*qw)
    r22 = 1 - 2 * (qx*qx + qy*qy)

    is_batched = qw.dim() > 0
    stack_dim = 1 if is_batched else 0

    return torch.stack([
        torch.stack([r00, r01, r02], dim=-1),
        torch.stack([r10, r11, r12], dim=-1),
        torch.stack([r20, r21, r22], dim=-1),
    ], dim=stack_dim).to(device)


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
    Rotate vector v from world frame to body frame: v_body = R(q)^T * v_world
    Supports both single (4,) and (3,) vectors, as well as batches (B, 4) and (B, 3).

    q_wxyz: (4,) or (B, 4)
    v: (3,) or (B, 3)
    """
    # We check if the input is batched by looking at the number of dimensions. If it's 2D, we assume it's (B, 4) and (B, 3). If it's 1D, we assume it's (4,) and (3,).
    is_batched = q_wxyz.dim() == 2

    # If it's a single example, artificially add a batch dimension (dimension 0)
    if not is_batched:
        q_wxyz = q_wxyz.unsqueeze(0)  # (4,) -> (1, 4)
        v = v.unsqueeze(0)            # (3,) -> (1, 3)

    # We take the quaternion components. For the batched case, this will give us tensors of shape (B,).
    qw, qx, qy, qz = q_wxyz[:, 0], q_wxyz[:, 1], q_wxyz[:, 2], q_wxyz[:, 3]
    
    # quat_to_rot also supports (B, 3, 3)
    R = quat_to_rot(qw, qx, qy, qz, device)  # (B, 3, 3)

    # Multiply R^T @ v using einsum for the whole batch.
    # 'bij' is matrix R (b-batch, i-row, j-column).
    # 'bi' is vector v (b-batch, i-element).
    # We sum over 'i' (rows of R), which mathematically corresponds to multiplication by the transpose (R^T).
    v_body = torch.einsum('bij,bi->bj', R, v)  # (B, 3)

    # If the input was a single vector, restore its original shape (3,)
    if not is_batched:
        return v_body.squeeze(0)
        
    return v_body
