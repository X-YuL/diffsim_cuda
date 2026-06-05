"""
Parity tests for the PyTorch vectorization of env.py / gait.py.

Background: srbd.py was already batched over the B parallel robots; these tests
cover the same treatment applied to the remaining per-env Python for-loops. They
verify the new batched implementations are numerically equivalent to the
original per-env loops they replace:

  1. estimate_foot_forces  (env.py)  -- batched (B,12,12) SVD vs per-env loop
  2. _mix_stance           (gait.py) -- batched top-k vs per-env loop
  3. autograd flow through the batched estimate_foot_forces is preserved

Self-contained: runs on CPU with synthetic data, no Isaac Gym / GPU required.
The batched bodies below mirror the implementations in env.py / gait.py; the
loop bodies are the originals they replaced. Run on the training machine
(PyTorch >= 1.10):

    python test_vectorization.py
"""
import math

import torch

torch.manual_seed(0)


class _Cfg:
    """Stub holding only the cfg fields these functions read."""
    pd_kp = 28.0
    pd_kd = 0.7
    fz_min = 0.0
    fz_max = 80.0
    mu_tangent = 0.6


# =====================================================================
# 1) estimate_foot_forces  (env.py)
# =====================================================================
def _foot_forces_loop(J12, tau_in, stance_mask, cfg, dtype):
    """Original per-env loop (env.py before vectorization), run in `dtype`."""
    B = J12.shape[0]
    dev = J12.device
    tau = tau_in.view(B, 12, 1).to(dtype)
    J12 = J12.to(dtype)
    sm = stance_mask.to(dtype)
    f_list = []
    for b in range(B):
        Jb = J12[b]
        wmask = sm[b].view(4, 1, 1)
        Jw = Jb * wmask
        Jbig = torch.cat([Jw[i] for i in range(4)], dim=0)  # (12,12)
        JJt = Jbig @ Jbig.T
        rhs = Jbig @ tau[b]
        U, S, Vh = torch.linalg.svd(JJt + 1e-9 * torch.eye(12, device=dev, dtype=dtype))
        S = torch.clamp(S, min=1e-3)
        Ainv = U @ torch.diag_embed(1.0 / S) @ Vh
        y = Ainv @ rhs
        f_b = -y.view(4, 3)
        fz = torch.clamp(f_b[:, 2:3], min=cfg.fz_min, max=cfg.fz_max) * sm[b]
        ft = f_b[:, :2]
        ft_norm = torch.linalg.norm(ft, dim=-1, keepdim=True)
        ft_max = cfg.mu_tangent * fz
        scale = torch.clamp(ft_max / (ft_norm + 1e-6), max=1.0)
        ft = ft * scale
        f_list.append(torch.cat([ft, fz], dim=-1))
    return torch.stack(f_list, dim=0)


def _foot_forces_batched(J12, tau_in, stance_mask, cfg):
    """Mirror of env.py:estimate_foot_forces (batched, float64 solve core)."""
    B = J12.shape[0]
    dev = J12.device
    tau = tau_in.view(B, 12, 1)
    Jw = J12 * stance_mask.view(B, 4, 1, 1)
    Jbig = Jw.reshape(B, 12, 12)
    JJt = (Jbig @ Jbig.transpose(-1, -2)).double()
    rhs = (Jbig @ tau).double()
    eye = torch.eye(12, device=dev, dtype=torch.float64)
    U, S, Vh = torch.linalg.svd(JJt + 1e-9 * eye)
    S = torch.clamp(S, min=1e-3)
    Ainv = U @ torch.diag_embed(1.0 / S) @ Vh
    y = Ainv @ rhs
    f = (-y.view(B, 4, 3)).float()
    fz = torch.clamp(f[..., 2:3], min=cfg.fz_min, max=cfg.fz_max) * stance_mask
    ft = f[..., :2]
    ft_norm = torch.linalg.norm(ft, dim=-1, keepdim=True)
    ft_max = cfg.mu_tangent * fz
    scale = torch.clamp(ft_max / (ft_norm + 1e-6), max=1.0)
    ft = ft * scale
    return torch.cat([ft, fz], dim=-1)


def test_estimate_foot_forces():
    B = 16
    cfg = _Cfg()
    J12 = torch.randn(B, 4, 3, 12, dtype=torch.float32)
    tau = torch.randn(B, 12, dtype=torch.float32)
    stance = (torch.rand(B, 4, 1) > 0.5).float()  # random subset of feet in stance

    f_batched = _foot_forces_batched(J12, tau, stance, cfg)
    f_loop64 = _foot_forces_loop(J12, tau, stance, cfg, torch.float64).float()
    f_loop32 = _foot_forces_loop(J12, tau, stance, cfg, torch.float32)

    d64 = (f_batched - f_loop64).abs().max().item()
    d32 = (f_batched - f_loop32).abs().max().item()
    print(f"[estimate_foot_forces] max|batched - loop(f64)| = {d64:.3e}")
    print(f"[estimate_foot_forces] max|batched - loop(f32)| = {d32:.3e}  (float32-loop rounding)")
    # Vectorization correctness: the batched float64 core must match the float64
    # per-env loop to solver tolerance (the SVD pseudo-inverse is sign-invariant).
    assert torch.allclose(f_batched, f_loop64, atol=1e-6, rtol=1e-5), \
        f"batched vs per-env loop (f64) mismatch: max|d|={d64:.3e}"
    print("[estimate_foot_forces] PASS (batched == per-env loop)\n")


def test_grad_flow():
    # Full stance => JJt full-rank => distinct singular values => stable SVD
    # backward; isolates "does autograd flow through the batched path" from the
    # (pre-existing) rank-deficient-SVD degeneracy.
    B = 4
    cfg = _Cfg()
    J12 = torch.randn(B, 4, 3, 12)
    stance = torch.ones(B, 4, 1)
    qref = torch.randn(B, 12, requires_grad=True)
    qnow = torch.randn(B, 12)
    qdnow = torch.randn(B, 12)
    tau = cfg.pd_kp * (qref - qnow) - cfg.pd_kd * qdnow
    f = _foot_forces_batched(J12, tau, stance, cfg)
    f.sum().backward()
    assert qref.grad is not None and torch.isfinite(qref.grad).all(), \
        "autograd did not flow finite grads to qref through batched estimate_foot_forces"
    print("[grad] PASS (qref.grad finite -> autograd preserved through batched SVD)\n")


# =====================================================================
# 2) _mix_stance  (gait.py)
# =====================================================================
def _phase_u(phases):
    return torch.remainder(phases, 2 * math.pi) / (2 * math.pi)


def _stance_phase_mask(phases, beta_B):
    B = phases.shape[0]
    u = _phase_u(phases)
    beta = beta_B.view(B, 1)
    return (u < beta).float().unsqueeze(-1)  # (B,4,1)


def _mix_stance_loop(phases, beta_B, min_feet_B):
    """Original per-env loop (gait.py before vectorization)."""
    B = phases.shape[0]
    mix = torch.clamp(_stance_phase_mask(phases, beta_B), 0.0, 1.0)
    flat = mix.view(B, 4).clone()
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


def _mix_stance_batched(phases, beta_B, min_feet_B):
    """Mirror of gait.py:_mix_stance (batched top-k via stable sort)."""
    B = phases.shape[0]
    dev = phases.device
    mix = torch.clamp(_stance_phase_mask(phases, beta_B), 0.0, 1.0)
    flat = mix.view(B, 4)
    min_feet_f = min_feet_B.to(flat.dtype)
    needs = flat.sum(dim=1) < min_feet_f
    _, order = torch.sort(flat, dim=1, descending=True, stable=True)
    rank = torch.zeros_like(order)
    rank.scatter_(1, order, torch.arange(4, device=dev).view(1, 4).expand(B, 4))
    forced = (rank < min_feet_B.view(B, 1)).to(flat.dtype)
    flat = torch.where(needs.view(B, 1), forced, flat)
    return flat.view(B, 4, 1)


def test_mix_stance():
    B = 256
    phases = torch.rand(B, 4) * 2 * math.pi
    beta_B = torch.rand(B) * 0.6 + 0.2  # in (0.2, 0.8)
    min_feet_B = torch.tensor([0, 2, 4])[torch.randint(0, 3, (B,))]

    res = _mix_stance_batched(phases, beta_B, min_feet_B).view(B, 4)
    ref = _mix_stance_loop(phases, beta_B, min_feet_B).view(B, 4)
    phase_mask = _stance_phase_mask(phases, beta_B).view(B, 4)
    orig_count = phase_mask.sum(1)

    # (1) original phase-stance feet are never dropped
    assert torch.equal(phase_mask.bool() & res.bool(), phase_mask.bool()), \
        "a real stance foot was dropped by the batched min-feet enforcement"
    # (2) support-foot count is correct: forced up to k, otherwise unchanged
    exp_count = torch.where(min_feet_B == 0, orig_count,
                            torch.maximum(orig_count, min_feet_B.float()))
    assert torch.equal(res.sum(1), exp_count), "min-support-feet count mismatch"
    print("[_mix_stance] PASS (semantics: stance preserved + correct support count)")

    # (3) exact bitwise parity incl. tie-break vs the per-env topk loop
    if torch.equal(res, ref):
        print("[_mix_stance] exact match incl. tie-break ordering vs per-env topk\n")
    else:
        n = int((res != ref).any(1).sum().item())
        print(f"[_mix_stance] NOTE: {n}/{B} envs differ only in which *swing* foot was "
              f"promoted among ties (arbitrary in the original); semantics identical.\n")


if __name__ == "__main__":
    print(f"torch {torch.__version__}\n")
    test_estimate_foot_forces()
    test_grad_flow()
    test_mix_stance()
    print("All vectorization parity tests passed.")
