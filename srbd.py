# ⚠️ CRITICAL: Isaac Gym must be imported BEFORE torch
try:
    from isaacgym import gymapi
except Exception:
    pass

import torch

from config import CUDA_KERNEL_SRBD
from utils_math import quat_rotate_inverse_wxyz, quat_to_rot

# ---------------------------------------------------------------------------
# CUDA kernel dispatch setup
#
# Controlled by CUDA_KERNEL_SRBD in config.py.
# Set to True to use the custom CUDA kernel for _srbd_step.
# Requires building the extension first: python setup.py build_ext --inplace
# Falls back to PyTorch automatically if the extension is not available.
# ---------------------------------------------------------------------------
_USE_CUDA_KERNEL = CUDA_KERNEL_SRBD
_srbd_cuda_ext = None

if _USE_CUDA_KERNEL:
    try:
        import srbd_cuda_ext as _srbd_cuda_ext
        print("[SRBD] Custom CUDA kernel active (CUDA_KERNEL_SRBD=True in config.py).")
    except ImportError as e:
        print(f"[SRBD] WARNING: CUDA_KERNEL_SRBD=True but extension not found ({e}).")
        print("[SRBD]          Falling back to PyTorch implementation.")
        print("[SRBD]          Run: python setup.py build_ext --inplace")
        _USE_CUDA_KERNEL = False


# ---------------------------------------------------------------------------
# Standalone PyTorch forward function for _srbd_step
#
# This is the same math as SRBDModel._srbd_step but expressed as a pure
# function (no class state).  It is used by:
#   1. SRBDStepFunction.backward  — re-run with grad tracking to get grads
#   2. (optionally) as a reference for numerical verification
# ---------------------------------------------------------------------------
def _srbd_step_pytorch_fn(p, v, q, w, f_world, q_ref12, m, g, Ixx, Iyy, Izz, dt, device):
    """Pure PyTorch SRBD step: inputs → (p_new, v_new, q_new, w_new)."""
    Fsum = f_world.sum(dim=1) + torch.tensor([0.0, 0.0, -m*g], device=device).view(1, 3)
    a = Fsum / m

    B = p.shape[0]
    L1, L2 = 0.213, 0.213
    hip_offsets = torch.tensor([
        [+0.1934, +0.1420, 0.0],
        [+0.1934, -0.1420, 0.0],
        [-0.1934, +0.1420, 0.0],
        [-0.1934, -0.1420, 0.0],
    ], device=device)

    R = quat_to_rot(q[:, 0], q[:, 1], q[:, 2], q[:, 3], device)  # (B,3,3)
    hip_world = p.unsqueeze(1) + torch.einsum('bij,nj->bni', R, hip_offsets)
    q_leg = q_ref12.view(B, 4, 3)
    q2, q3 = q_leg[:, :, 1], q_leg[:, :, 2]
    ox = L1*torch.sin(q2) + L2*torch.sin(q2 + q3)
    oz = -L1*torch.cos(q2) - L2*torch.cos(q2 + q3)
    off_world = torch.einsum('bij,bnj->bni', R,
                             torch.stack([ox, torch.zeros_like(ox), oz], dim=-1))
    p_foot = hip_world + off_world  # (B,4,3)

    r = p_foot - p.unsqueeze(1)
    tau_world = torch.cross(r, f_world, dim=-1).sum(dim=1)
    tau_body = torch.einsum('bji,bj->bi', R, tau_world)

    I_diag = torch.tensor([Ixx, Iyy, Izz], device=device)
    Iw = w * I_diag.unsqueeze(0)
    wdot = (tau_body - torch.cross(w, Iw, dim=-1)) / I_diag.unsqueeze(0)
    w_new = w + wdot * dt

    wx, wy, wz = w_new.unbind(dim=-1)
    z = torch.zeros_like(wx)
    Omega = torch.stack([
        torch.stack([z,  -wx, -wy, -wz], dim=-1),
        torch.stack([wx,  z,   wz, -wy], dim=-1),
        torch.stack([wy, -wz,  z,   wx], dim=-1),
        torch.stack([wz,  wy, -wx,  z ], dim=-1),
    ], dim=1)
    qdot = 0.5 * torch.einsum('bij,bj->bi', Omega, q)
    q_new = q + qdot * dt
    q_new = q_new / (q_new.norm(dim=-1, keepdim=True) + 1e-9)

    return p + v*dt, v + a*dt, q_new, w_new


# ---------------------------------------------------------------------------
# SRBDStepFunction — torch.autograd.Function wrapper
#
# Forward: calls the fast CUDA kernel (srbd_step_kernel).
# Backward: re-runs the same computation in PyTorch with gradient tracking
#           and uses torch.autograd.grad to recover gradients.  This avoids
#           writing manual derivative formulas while preserving correctness.
#
# All six tensor inputs (p, v, q, w, f_world, q_ref12) carry gradients.
# This is required because within a training iteration the SRBD state
# (srbd_p/v/q/w) carries grad_fn chains back into previous steps' policy
# outputs (the strict alpha-alignment at train.py:188-192 deliberately
# preserves these chains with multiplier alpha).  Detachment only happens
# at the iteration boundary (train.py:415-418), not at every step.
# ---------------------------------------------------------------------------
class SRBDStepFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, v, q, w, f_world, q_ref12, m, g, Ixx, Iyy, Izz, dt):
        ctx.save_for_backward(p, v, q, w, f_world, q_ref12)
        ctx.physics = (m, g, Ixx, Iyy, Izz, dt)
        ctx.dev = f_world.device

        p_new, v_new, q_new, w_new = _srbd_cuda_ext.srbd_step_forward(
            p.contiguous(), v.contiguous(), q.contiguous(), w.contiguous(),
            f_world.contiguous(), q_ref12.contiguous(),
            m, g, Ixx, Iyy, Izz, dt
        )
        return p_new, v_new, q_new, w_new

    @staticmethod
    def backward(ctx, grad_p_new, grad_v_new, grad_q_new, grad_w_new):
        p, v, q, w, f_world, q_ref12 = ctx.saved_tensors
        m, g, Ixx, Iyy, Izz, dt = ctx.physics
        dev = ctx.dev

        # ctx.needs_input_grad: tuple of bools, one per forward() input.
        # Slots 0..5 are the six tensor inputs (p, v, q, w, f_world, q_ref12);
        # slots 6..11 are scalars (always None grad).
        needs = ctx.needs_input_grad[:6]

        # Short-circuit when no input needs grad.  In practice PyTorch will
        # not call backward in that case, but the guard keeps torch.autograd.grad
        # from receiving an empty `inputs` list.
        if not any(needs):
            return (None,) * 12

        # Build a fresh leaf tensor per input with requires_grad=True so the
        # re-run constructs a fully differentiable graph.  We later mask the
        # returned grads with None for slots the upstream graph doesn't need.
        p_t  = p.detach().requires_grad_(True)
        v_t  = v.detach().requires_grad_(True)
        q_t  = q.detach().requires_grad_(True)
        w_t  = w.detach().requires_grad_(True)
        f_t  = f_world.detach().requires_grad_(True)
        qr_t = q_ref12.detach().requires_grad_(True)

        with torch.enable_grad():
            outputs = _srbd_step_pytorch_fn(
                p_t, v_t, q_t, w_t, f_t, qr_t,
                m, g, Ixx, Iyy, Izz, dt, dev
            )

        # Only request grads for inputs the upstream graph actually needs.
        # Passing a tensor with requires_grad=False (or omitted) here is what
        # tripped the original "Tensor does not require grad" error.
        inputs_all = [p_t, v_t, q_t, w_t, f_t, qr_t]
        wanted = [t for t, n in zip(inputs_all, needs) if n]

        grads_wanted = torch.autograd.grad(
            outputs=list(outputs),
            inputs=wanted,
            grad_outputs=[grad_p_new, grad_v_new, grad_q_new, grad_w_new],
            only_inputs=True,
            allow_unused=True,
        )

        it = iter(grads_wanted)
        g_p, g_v, g_q, g_w, g_f, g_qr = [next(it) if n else None for n in needs]

        # Gradient slots: p, v, q, w, f_world, q_ref12, m, g, Ixx, Iyy, Izz, dt
        return (g_p, g_v, g_q, g_w, g_f, g_qr,
                None, None, None, None, None, None)


# ---------------------------------------------------------------------------
# SRBDModel
#
# Differentiable Single Rigid Body Dynamics proxy used during training.
# Holds no state of its own — it proxies attribute access onto the wrapped
# `env` via __getattr__ / __setattr__, so `self.srbd_p`, `self.srbd_q`,
# `self.cfg`, `self.device`, etc. all resolve to `env.srbd_p`, etc.
#
# The hot path is `_srbd_step(...)`, which dispatches to either:
#   - the custom CUDA kernel (when CUDA_KERNEL_SRBD=True in config.py and
#     the extension is built), wrapped by SRBDStepFunction for autograd, or
#   - a pure PyTorch implementation (default — always available).
# Both paths produce numerically equivalent forward outputs and matching
# gradients; see the SRBDStepFunction docstring above for details.
# ---------------------------------------------------------------------------

class SRBDModel:
    def __init__(self, env):
        object.__setattr__(self, 'env', env)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def __setattr__(self, name, value):
        if name == 'env':
            object.__setattr__(self, name, value)
        else:
            setattr(self.env, name, value)

    def _srbd_init_from_isaac(self):
        """Initialize full 3D SRBD state from Isaac (batch)."""
        dev = self.device
        self.srbd_p = self.base_pos.clone()            # (B,3)
        self.srbd_v = self.base_lin_world.clone()      # (B,3)
        self.srbd_q = self.base_quat.clone()           # (B,4)

        self.srbd_w = quat_rotate_inverse_wxyz(self.srbd_q, self.base_ang_world, dev) # (B,3)

    # Normalize quaternion to unit length

    def _quat_norm(self, q):
        if q.dim() == 1:
            return q / (q.norm() + 1e-9)
        else:
            return q / (q.norm(dim=-1, keepdim=True) + 1e-9)


    def _srbd_step(self, f_world, q_ref12, dt):
        """
        One Euler integration step of the centroidal SRBD dynamics (batch).

        Reads  self.srbd_p, self.srbd_v, self.srbd_q (wxyz), self.srbd_w (body frame)
        Writes self.srbd_p, self.srbd_v, self.srbd_q, self.srbd_w  (in place via the
               SRBDModel proxy, i.e. on the underlying env)

        Args:
            f_world : (B, 4, 3) ground-reaction forces per foot, world frame
            q_ref12 : (B, 12)   reference joint angles (hip, thigh, calf) × 4 legs
            dt      : float     integration step (e.g. cfg.dt = 0.002)

        Dispatch:
            CUDA_KERNEL_SRBD=True  -> SRBDStepFunction.apply (fused CUDA kernel +
                                       autograd wrapper with full gradient flow).
            CUDA_KERNEL_SRBD=False -> the pure PyTorch implementation below
                                       (default; autograd flows natively).
        Both paths produce equivalent results to float32 precision.
        """
        if _USE_CUDA_KERNEL:
            # ---- CUDA kernel path ----
            p_new, v_new, q_new, w_new = SRBDStepFunction.apply(
                self.srbd_p.contiguous(),
                self.srbd_v.contiguous(),
                self.srbd_q.contiguous(),
                self.srbd_w.contiguous(),
                f_world,
                q_ref12,
                float(self.cfg.m),
                float(self.cfg.g),
                float(self.cfg.Ixx),
                float(self.cfg.Iyy),
                float(self.cfg.Izz),
                float(dt),
            )
            self.srbd_p = p_new
            self.srbd_v = v_new
            self.srbd_q = q_new
            self.srbd_w = w_new
        else:
            # ---- Original PyTorch path (unchanged) ----
            dev = self.device
            m, g = self.cfg.m, self.cfg.g

            # Snapshot of current SRBD state (don't modify self.srbd_* midway)
            p = self.srbd_p          # (B,3)
            v = self.srbd_v          # (B,3)
            q = self.srbd_q          # (B,4)
            w = self.srbd_w          # (B,3)

            # ---- Translational part ----
            Fsum = f_world.sum(dim=1) + torch.tensor(
                [0.0, 0.0, -m * g], device=dev
            ).view(1, 3)                               # (B,3)
            a = Fsum / m                                # (B,3)


            # First use "old state" to calculate foot positions and torques
            #p_foot = self.foot_positions_srbd(q_ref12.detach())  # (B,4,3)
            p_foot = self.foot_positions_srbd(q_ref12)

            # Force arm vectors r: from COM to feet for the whole batch
            r = p_foot - p.unsqueeze(1)                 # (B,4,3) -> automatic broadcasting

            # Batched cross product (world-frame torque)
            tau_world = torch.cross(r, f_world, dim=-1).sum(dim=1)  # (B,3)

            # Generate rotation matrices R for the whole batch at once
            R = quat_to_rot(q[:, 0], q[:, 1], q[:, 2], q[:, 3], dev) # (B,3,3)

            # world -> body: equivalent to R.t() @ tau_world for each batch element
            tau_body = torch.einsum('bji,bj->bi', R, tau_world) # (B,3)

            # Since I is diagonal, treat it as a (3,) vector and use fast broadcasting
            I_diag = torch.tensor([self.cfg.Ixx, self.cfg.Iyy, self.cfg.Izz], device=dev) # (3,)

            # Euler equation: I * wdot = tau - w × (I w) without the loop
            Iw = w * I_diag.unsqueeze(0)                # (B,3) - fast elementwise multiplication
            w_cross_Iw = torch.cross(w, Iw, dim=-1) # (B,3)
            wdot = (tau_body - w_cross_Iw) / I_diag.unsqueeze(0) # (B,3)

            w_new = w + wdot * dt                      # (B,3)

            # Quaternion update using batch operations
            wx, wy, wz = w_new.unbind(dim=-1)           # each has shape (B,)
            z = torch.zeros_like(wx)                    # (B,)

            # Construct batch Omega tensor (B, 4, 4)
            Omega = torch.stack([
                torch.stack([z,  -wx, -wy, -wz], dim=-1),
                torch.stack([wx,  z,   wz, -wy], dim=-1),
                torch.stack([wy, -wz,  z,   wx], dim=-1),
                torch.stack([wz,  wy, -wx,  z ], dim=-1),
            ], dim=1)                                   # (B,4,4)

            # qdot = 0.5 * (Omega @ q) for the whole batch
            qdot = 0.5 * torch.einsum('bij,bj->bi', Omega, q) # (B,4)
            q_new = self._quat_norm(q + qdot * dt)      # (B,4)

            # ---- State update ----
            self.srbd_p = p + v * dt                    # (B,3)
            self.srbd_v = v + a * dt                    # (B,3)
            self.srbd_q = q_new                         # (B,4)
            self.srbd_w = w_new                         # (B,3)



    # ---------------- SRBD-based foot positions ----------------

    def foot_positions_srbd(self, q_ref12):
        """
        3D SRBD foot positions (batch)
        q_ref12: (B,12)
        Returns (B,4,3)
        """
        dev = self.device
        B = self.B
        p_base = self.srbd_p          # (B,3)
        q = self.srbd_q               # (B,4)

        hip_offsets = torch.tensor([
            [ +0.1934, +0.1420, 0.0 ], # FL, FR, RL, RR
            [ +0.1934, -0.1420, 0.0 ],
            [ -0.1934, +0.1420, 0.0 ],
            [ -0.1934, -0.1420, 0.0 ],
        ], device=dev)                # (4,3)

        #L1, L2 = 0.25, 0.25
        L1, L2 = 0.213, 0.213           # this should be in the env/config --> change later on

        # Get rotation matrices for the whole batch
        R = quat_to_rot(q[:, 0], q[:, 1], q[:, 2], q[:, 3], dev) # (B,3,3)

        # Project hip_offsets from body to world using einsum
        # (B,3,3) @ (4,3) -> (B,4,3)
        hip_world = p_base.unsqueeze(1) + torch.einsum('bij,nj->bni', R, hip_offsets) # (B,4,3)

        # Leg kinematics for the whole batch
        q_leg = q_ref12.view(B, 4, 3)               # (B,4,3)
        q2, q3 = q_leg[:, :, 1], q_leg[:, :, 2]     # (B,4)

        x = L1 * torch.sin(q2) + L2 * torch.sin(q2 + q3) # (B,4)
        z = -L1 * torch.cos(q2) - L2 * torch.cos(q2 + q3) # (B,4)
        y = torch.zeros_like(x)                     # (B,4)

        off_body = torch.stack([x, y, z], dim=-1)  # (B,4,3)

        # Rotate foot positions from body to world for the whole batch
        off_world = torch.einsum('bij,bnj->bni', R, off_body) # (B,4,3)

        return hip_world + off_world                # (B,4,3)


    # ---------------- env step ----------------
