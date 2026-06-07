"""
SRBD CUDA kernel parity test.

Verifies that:
  1. srbd_step_forward (CUDA) matches the inline PyTorch reference path
     (the one in srbd.py:_srbd_step under `else:`) to within ~1e-4 abs.
  2. srbd_step_backward (CUDA) matches torch.autograd.backward on that
     same PyTorch reference to within ~1e-3 abs.
  3. SRBDStepFunction.apply + .backward() (the autograd wrapper) wires
     both directions correctly end-to-end.

Tolerances are deliberately loose: the kernel is built with --use_fast_math
(sinf/cosf/rsqrtf ~1-2 ULP), and PyTorch uses standard math. See the
"Testing the kernel" section in README.md.

Run after every rebuild:
    python setup.py build_ext --inplace
    python tests/test_srbd_kernel.py
"""

import os
import sys

# This test lives in tests/ but imports project modules (srbd, utils_math) and
# the built srbd_cuda_ext extension, which sit in the repo root. Put the repo
# root (one level up) on sys.path so it works both as a direct script
# (python tests/test_srbd_kernel.py) and under pytest.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import srbd_cuda_ext as _ext

# Ensure srbd.SRBDStepFunction can dispatch into the extension regardless of
# the CUDA_KERNEL_SRBD flag in config.py — this test is about the kernel, not
# the dispatch toggle.
import srbd as _srbd_mod
_srbd_mod._srbd_cuda_ext = _ext
from srbd import SRBDStepFunction  # noqa: E402

from utils_math import quat_to_rot  # noqa: E402


DEV = "cuda"
DTY = torch.float32

# Physics constants (mirror config.EnvCfg defaults).
M_MASS, G_GRAV = 7.0, 9.81
IXX, IYY, IZZ = 0.024, 0.098, 0.107
DT = 0.002

# Robot geometry (mirror srbd_cuda.cu and foot_positions_srbd).
L1, L2 = 0.213, 0.213
HIP_OFFSETS = torch.tensor([
    [+0.1934, +0.1420, 0.0],
    [+0.1934, -0.1420, 0.0],
    [-0.1934, +0.1420, 0.0],
    [-0.1934, -0.1420, 0.0],
], device=DEV, dtype=DTY)


def srbd_step_pytorch(p, v, q, w, f_world, q_ref12,
                      m, g, Ixx, Iyy, Izz, dt):
    """Inline copy of SRBDModel._srbd_step's `else:` branch (the PyTorch path).

    Must mirror srbd.py:183-235 exactly so the test's reference is the same
    code path that runs when CUDA_KERNEL_SRBD=False.
    """
    B = p.shape[0]
    dev = p.device

    Fsum = f_world.sum(dim=1) + torch.tensor([0.0, 0.0, -m * g], device=dev).view(1, 3)
    a = Fsum / m

    R = quat_to_rot(q[:, 0], q[:, 1], q[:, 2], q[:, 3], dev)  # (B, 3, 3)
    hip_world = p.unsqueeze(1) + torch.einsum('bij,nj->bni', R, HIP_OFFSETS)
    q_leg = q_ref12.view(B, 4, 3)
    q2, q3 = q_leg[:, :, 1], q_leg[:, :, 2]
    ox = L1 * torch.sin(q2) + L2 * torch.sin(q2 + q3)
    oz = -L1 * torch.cos(q2) - L2 * torch.cos(q2 + q3)
    oy = torch.zeros_like(ox)
    off_body = torch.stack([ox, oy, oz], dim=-1)
    off_world = torch.einsum('bij,bnj->bni', R, off_body)
    p_foot = hip_world + off_world

    r = p_foot - p.unsqueeze(1)
    tau_world = torch.cross(r, f_world, dim=-1).sum(dim=1)
    tau_body = torch.einsum('bji,bj->bi', R, tau_world)

    I_diag = torch.tensor([Ixx, Iyy, Izz], device=dev)
    Iw = w * I_diag.unsqueeze(0)
    w_cross_Iw = torch.cross(w, Iw, dim=-1)
    wdot = (tau_body - w_cross_Iw) / I_diag.unsqueeze(0)
    w_new = w + wdot * dt

    wx, wy, wz = w_new.unbind(dim=-1)
    zr = torch.zeros_like(wx)
    Omega = torch.stack([
        torch.stack([zr,  -wx, -wy, -wz], dim=-1),
        torch.stack([wx,  zr,   wz, -wy], dim=-1),
        torch.stack([wy, -wz,  zr,   wx], dim=-1),
        torch.stack([wz,  wy, -wx,  zr], dim=-1),
    ], dim=1)
    qdot = 0.5 * torch.einsum('bij,bj->bi', Omega, q)
    q_unnorm = q + qdot * dt
    q_new = q_unnorm / (q_unnorm.norm(dim=-1, keepdim=True) + 1e-9)

    p_new = p + v * dt
    v_new = v + a * dt
    return p_new, v_new, q_new, w_new


def make_inputs(B=16, seed=0):
    """Realistic-range random inputs. Generator is local so callers don't
    perturb global RNG state used elsewhere in the test."""
    rng = torch.Generator(device=DEV).manual_seed(seed)
    p = torch.randn(B, 3, generator=rng, device=DEV, dtype=DTY)
    v = torch.randn(B, 3, generator=rng, device=DEV, dtype=DTY)
    q_raw = torch.randn(B, 4, generator=rng, device=DEV, dtype=DTY)
    q = q_raw / q_raw.norm(dim=-1, keepdim=True)
    w = torch.randn(B, 3, generator=rng, device=DEV, dtype=DTY) * 0.5
    f_world = torch.randn(B, 4, 3, generator=rng, device=DEV, dtype=DTY)
    q_ref12 = torch.randn(B, 12, generator=rng, device=DEV, dtype=DTY) * 0.3
    return p, v, q, w, f_world, q_ref12


def report(name, cuda_t, ref_t, atol, rtol):
    diff = (cuda_t - ref_t).abs()
    rel = diff / ref_t.abs().clamp_min(1e-9)
    ok = torch.allclose(cuda_t, ref_t, atol=atol, rtol=rtol)
    tag = "OK " if ok else "FAIL"
    print(f"  [{tag}] {name:<12} max abs {diff.max().item():.2e}  max rel {rel.max().item():.2e}")
    return ok


def test_forward():
    print("[1/3] Forward parity (atol=1e-4, rtol=1e-3)")
    p, v, q, w, fw, qr = make_inputs()

    pn_cu, vn_cu, qn_cu, wn_cu = _ext.srbd_step_forward(
        p.contiguous(), v.contiguous(), q.contiguous(), w.contiguous(),
        fw.contiguous(), qr.contiguous(),
        M_MASS, G_GRAV, IXX, IYY, IZZ, DT)
    pn_pt, vn_pt, qn_pt, wn_pt = srbd_step_pytorch(
        p.clone(), v.clone(), q.clone(), w.clone(), fw.clone(), qr.clone(),
        M_MASS, G_GRAV, IXX, IYY, IZZ, DT)

    all_ok = True
    for name, cu, pt in [("p_new", pn_cu, pn_pt), ("v_new", vn_cu, vn_pt),
                         ("q_new", qn_cu, qn_pt), ("w_new", wn_cu, wn_pt)]:
        all_ok &= report(name, cu, pt, atol=1e-4, rtol=1e-3)
    assert all_ok, "Forward parity failed — see max-diff numbers above."


def test_backward_direct():
    print("[2/3] Backward parity, direct ext call (atol=1e-3, rtol=1e-2)")
    torch.manual_seed(0)
    p, v, q, w, fw, qr = make_inputs()

    p_pt, v_pt, q_pt, w_pt, fw_pt, qr_pt = [
        t.detach().clone().requires_grad_(True) for t in (p, v, q, w, fw, qr)
    ]
    pn, vn, qn, wn = srbd_step_pytorch(
        p_pt, v_pt, q_pt, w_pt, fw_pt, qr_pt,
        M_MASS, G_GRAV, IXX, IYY, IZZ, DT)

    g_p_new = torch.randn_like(pn)
    g_v_new = torch.randn_like(vn)
    g_q_new = torch.randn_like(qn)
    g_w_new = torch.randn_like(wn)
    torch.autograd.backward([pn, vn, qn, wn],
                            [g_p_new, g_v_new, g_q_new, g_w_new])

    g_p_cu, g_v_cu, g_q_cu, g_w_cu, g_f_cu, g_qr_cu = _ext.srbd_step_backward(
        p.contiguous(), v.contiguous(), q.contiguous(), w.contiguous(),
        fw.contiguous(), qr.contiguous(),
        g_p_new.contiguous(), g_v_new.contiguous(),
        g_q_new.contiguous(), g_w_new.contiguous(),
        M_MASS, G_GRAV, IXX, IYY, IZZ, DT)

    all_ok = True
    for name, cu, pt in [
        ("g_p",        g_p_cu,  p_pt.grad),
        ("g_v",        g_v_cu,  v_pt.grad),
        ("g_q",        g_q_cu,  q_pt.grad),
        ("g_w",        g_w_cu,  w_pt.grad),
        ("g_f_world",  g_f_cu,  fw_pt.grad),
        ("g_q_ref12",  g_qr_cu, qr_pt.grad),
    ]:
        all_ok &= report(name, cu, pt, atol=1e-3, rtol=1e-2)
    assert all_ok, "Backward parity failed — see max-diff numbers above."


def test_autograd_wrapper():
    print("[3/3] SRBDStepFunction.apply round-trip (atol=1e-3, rtol=1e-2)")
    torch.manual_seed(0)
    p, v, q, w, fw, qr = make_inputs()

    # CUDA path via autograd wrapper.
    p_cu, v_cu, q_cu, w_cu, fw_cu, qr_cu = [
        t.detach().clone().requires_grad_(True) for t in (p, v, q, w, fw, qr)
    ]
    pn, vn, qn, wn = SRBDStepFunction.apply(
        p_cu, v_cu, q_cu, w_cu, fw_cu, qr_cu,
        float(M_MASS), float(G_GRAV), float(IXX), float(IYY), float(IZZ), float(DT))

    g_p_new = torch.randn_like(pn)
    g_v_new = torch.randn_like(vn)
    g_q_new = torch.randn_like(qn)
    g_w_new = torch.randn_like(wn)
    torch.autograd.backward([pn, vn, qn, wn],
                            [g_p_new, g_v_new, g_q_new, g_w_new])

    # PyTorch reference for comparison.
    p_pt, v_pt, q_pt, w_pt, fw_pt, qr_pt = [
        t.detach().clone().requires_grad_(True) for t in (p, v, q, w, fw, qr)
    ]
    pn_pt, vn_pt, qn_pt, wn_pt = srbd_step_pytorch(
        p_pt, v_pt, q_pt, w_pt, fw_pt, qr_pt,
        M_MASS, G_GRAV, IXX, IYY, IZZ, DT)
    torch.autograd.backward([pn_pt, vn_pt, qn_pt, wn_pt],
                            [g_p_new, g_v_new, g_q_new, g_w_new])

    all_ok = True
    for name, cu, pt in [
        ("g_p",        p_cu.grad,  p_pt.grad),
        ("g_v",        v_cu.grad,  v_pt.grad),
        ("g_q",        q_cu.grad,  q_pt.grad),
        ("g_w",        w_cu.grad,  w_pt.grad),
        ("g_f_world",  fw_cu.grad, fw_pt.grad),
        ("g_q_ref12",  qr_cu.grad, qr_pt.grad),
    ]:
        all_ok &= report(name, cu, pt, atol=1e-3, rtol=1e-2)
    assert all_ok, "Autograd wrapper round-trip failed — see max-diff numbers above."


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available — this test requires a GPU.")
    test_forward()
    test_backward_direct()
    test_autograd_wrapper()
    print("\nAll SRBD kernel tests passed.")
