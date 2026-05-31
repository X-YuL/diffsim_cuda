/*
 * srbd_cuda.cu — Custom CUDA kernels for the SRBD (Single Rigid Body Dynamics) model.
 *
 * Two kernels are provided:
 *   1. foot_positions_kernel  — computes world-frame foot positions from SRBD state + joint angles
 *   2. srbd_step_kernel       — fused kernel: foot FK + Newton-Euler dynamics + quaternion integration
 *
 * Thread layout:
 *   Both kernels: grid=((B+255)/256,)  block=(256,)   one thread per environment.
 *   Per-env work is ~200 scalar FLOPs of straight-line math with no inter-env
 *   dependencies — no shared memory, no __syncthreads, no atomics needed.
 *
 * All tensors are float32, stored in row-major (C) order.
 * Requires CUDA compute capability >= 6.0 (Pascal).
 */

#include <cuda_runtime.h>
#include <math.h>
#include <torch/extension.h>
#include <vector>

// ---------------------------------------------------------------------------
// Robot geometry constants — stored in GPU constant memory (read-only, cached)
// Order: FL, FR, RL, RR  (matches env.py / srbd.py convention)
// ---------------------------------------------------------------------------
__constant__ float c_hip_offsets[4][3] = {
    {+0.1934f, +0.1420f, 0.0f},
    {+0.1934f, -0.1420f, 0.0f},
    {-0.1934f, +0.1420f, 0.0f},
    {-0.1934f, -0.1420f, 0.0f},
};
static constexpr float L1 = 0.213f;
static constexpr float L2 = 0.213f;

// ---------------------------------------------------------------------------
// Device helper: compute 3x3 rotation matrix from wxyz quaternion
// Writes 9 floats to R[9] in row-major order: R[row*3 + col]
// ---------------------------------------------------------------------------
__device__ __forceinline__ void quat_to_R(
    float qw, float qx, float qy, float qz,
    float R[9]
) {
    R[0] = 1.0f - 2.0f*(qy*qy + qz*qz);
    R[1] = 2.0f*(qx*qy - qz*qw);
    R[2] = 2.0f*(qx*qz + qy*qw);
    R[3] = 2.0f*(qx*qy + qz*qw);
    R[4] = 1.0f - 2.0f*(qx*qx + qz*qz);
    R[5] = 2.0f*(qy*qz - qx*qw);
    R[6] = 2.0f*(qx*qz - qy*qw);
    R[7] = 2.0f*(qy*qz + qx*qw);
    R[8] = 1.0f - 2.0f*(qx*qx + qy*qy);
}

// ---------------------------------------------------------------------------
// Kernel 1: foot_positions_kernel
//
// Computes world-frame foot positions from SRBD body pose + joint angles.
//
// Grid : ((B+255)/256,)   Block: (256,)   — one thread per environment
//
// Inputs  (row-major):
//   p_base  [B, 3]   — body position in world frame
//   q       [B, 4]   — body orientation quaternion (wxyz)
//   q_ref12 [B, 12]  — reference joint angles (hip, thigh, calf) x 4 legs
//
// Output:
//   p_foot  [B, 4, 3] — foot positions in world frame
// ---------------------------------------------------------------------------
__global__ void foot_positions_kernel(
    const float* __restrict__ p_base,   // [B, 3]
    const float* __restrict__ q,         // [B, 4]
    const float* __restrict__ q_ref12,   // [B, 12]
    float*       __restrict__ p_foot,    // [B, 4, 3]
    int B
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    // Build rotation matrix R (body -> world) for this env (local registers).
    float qw = q[b*4 + 0];
    float qx = q[b*4 + 1];
    float qy = q[b*4 + 2];
    float qz = q[b*4 + 3];
    float R[9];
    quat_to_R(qw, qx, qy, qz, R);

    float px = p_base[b*3 + 0];
    float py = p_base[b*3 + 1];
    float pz = p_base[b*3 + 2];

    // Compute all four foot positions sequentially.
    #pragma unroll
    for (int f = 0; f < 4; ++f) {
        // Hip world position: p_base + R · hip_offsets[f]
        float hx = px + R[0]*c_hip_offsets[f][0] + R[1]*c_hip_offsets[f][1] + R[2]*c_hip_offsets[f][2];
        float hy = py + R[3]*c_hip_offsets[f][0] + R[4]*c_hip_offsets[f][1] + R[5]*c_hip_offsets[f][2];
        float hz = pz + R[6]*c_hip_offsets[f][0] + R[7]*c_hip_offsets[f][1] + R[8]*c_hip_offsets[f][2];

        // 2-link planar leg FK (thigh, calf in sagittal plane; oy = 0)
        float q2 = q_ref12[b*12 + f*3 + 1];
        float q3 = q_ref12[b*12 + f*3 + 2];
        float ox = L1*sinf(q2) + L2*sinf(q2 + q3);
        float oz = -L1*cosf(q2) - L2*cosf(q2 + q3);

        // Rotate offset into world (oy = 0 -> only cols 0 and 2 of R used)
        p_foot[b*12 + f*3 + 0] = hx + R[0]*ox + R[2]*oz;
        p_foot[b*12 + f*3 + 1] = hy + R[3]*ox + R[5]*oz;
        p_foot[b*12 + f*3 + 2] = hz + R[6]*ox + R[8]*oz;
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: srbd_step_kernel
//
// Fused kernel: foot FK + centroidal dynamics + quaternion integration.
// Implements one Euler integration step of the SRBD equations of motion.
//
// Grid : ((B+255)/256,)   Block: (256,)   — one thread per environment
//
// Each thread runs the full per-env step sequentially in registers:
// build R, walk the 4 feet (FK + force/torque accumulation), then translational
// accel, Euler equation, quaternion update + normalize, and output writes.
//
// Inputs  (row-major):
//   p       [B, 3]    srbd_p  — body CoM position (world)
//   v       [B, 3]    srbd_v  — body CoM velocity (world)
//   q       [B, 4]    srbd_q  — body orientation quaternion (wxyz)
//   w       [B, 3]    srbd_w  — body angular velocity (body frame)
//   f_world [B, 4, 3] — ground reaction forces per foot (world frame)
//   q_ref12 [B, 12]   — reference joint angles (for foot FK)
//   m, g              — total mass and gravity magnitude
//   Ixx, Iyy, Izz     — principal inertia values (diagonal I matrix, body frame)
//   dt                — integration timestep
//
// Outputs (row-major):
//   p_new [B, 3]    — updated position
//   v_new [B, 3]    — updated velocity
//   q_new [B, 4]    — updated quaternion (normalized)
//   w_new [B, 3]    — updated angular velocity (body frame)
// ---------------------------------------------------------------------------
__global__ void srbd_step_kernel(
    const float* __restrict__ p,         // [B, 3]
    const float* __restrict__ v,         // [B, 3]
    const float* __restrict__ q,         // [B, 4]
    const float* __restrict__ w,         // [B, 3]
    const float* __restrict__ f_world,   // [B, 4, 3]
    const float* __restrict__ q_ref12,   // [B, 12]
    float*       __restrict__ p_new,     // [B, 3]
    float*       __restrict__ v_new,     // [B, 3]
    float*       __restrict__ q_new,     // [B, 4]
    float*       __restrict__ w_new,     // [B, 3]
    float m, float g,
    float Ixx, float Iyy, float Izz,
    float dt,
    int B
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    // ---- Build rotation matrix R (body -> world) in registers ----
    float qw = q[b*4 + 0];
    float qx = q[b*4 + 1];
    float qy = q[b*4 + 2];
    float qz = q[b*4 + 3];
    float R[9];
    quat_to_R(qw, qx, qy, qz, R);

    // ---- Read CoM position once ----
    float px = p[b*3 + 0];
    float py = p[b*3 + 1];
    float pz = p[b*3 + 2];

    // ---- Force/torque accumulators (gravity preloaded) ----
    float fsum_x = 0.0f;
    float fsum_y = 0.0f;
    float fsum_z = -m * g;
    float tau_x = 0.0f;
    float tau_y = 0.0f;
    float tau_z = 0.0f;

    // ---- Sequential per-foot loop: FK + force/torque accumulation ----
    #pragma unroll
    for (int f = 0; f < 4; ++f) {
        // Hip world position
        float hx = px + R[0]*c_hip_offsets[f][0] + R[1]*c_hip_offsets[f][1] + R[2]*c_hip_offsets[f][2];
        float hy = py + R[3]*c_hip_offsets[f][0] + R[4]*c_hip_offsets[f][1] + R[5]*c_hip_offsets[f][2];
        float hz = pz + R[6]*c_hip_offsets[f][0] + R[7]*c_hip_offsets[f][1] + R[8]*c_hip_offsets[f][2];

        // 2-link planar leg FK (oy = 0)
        float q2 = q_ref12[b*12 + f*3 + 1];
        float q3 = q_ref12[b*12 + f*3 + 2];
        float ox = L1*sinf(q2) + L2*sinf(q2 + q3);
        float oz = -L1*cosf(q2) - L2*cosf(q2 + q3);

        // Foot world position
        float fxw = hx + R[0]*ox + R[2]*oz;
        float fyw = hy + R[3]*ox + R[5]*oz;
        float fzw = hz + R[6]*ox + R[8]*oz;

        // Ground reaction force in world frame
        float fx = f_world[b*12 + f*3 + 0];
        float fy = f_world[b*12 + f*3 + 1];
        float fz = f_world[b*12 + f*3 + 2];

        fsum_x += fx;
        fsum_y += fy;
        fsum_z += fz;

        // Moment arm r = p_foot - p_com; tau += r × f
        float rx = fxw - px;
        float ry = fyw - py;
        float rz = fzw - pz;

        tau_x += ry*fz - rz*fy;
        tau_y += rz*fx - rx*fz;
        tau_z += rx*fy - ry*fx;
    }

    // ---- Translational acceleration ----
    float ax = fsum_x / m;
    float ay = fsum_y / m;
    float az = fsum_z / m;

    // ---- tau_body = R^T @ tau_world  (R^T[i,j] = R[j*3+i]) ----
    float tb0 = R[0]*tau_x + R[3]*tau_y + R[6]*tau_z;
    float tb1 = R[1]*tau_x + R[4]*tau_y + R[7]*tau_z;
    float tb2 = R[2]*tau_x + R[5]*tau_y + R[8]*tau_z;

    // ---- Euler's equation: wdot = (tau_body - w × (I·w)) / I ----
    float wb0 = w[b*3 + 0];
    float wb1 = w[b*3 + 1];
    float wb2 = w[b*3 + 2];

    float Iw0 = Ixx * wb0;
    float Iw1 = Iyy * wb1;
    float Iw2 = Izz * wb2;

    float wIw0 = wb1*Iw2 - wb2*Iw1;
    float wIw1 = wb2*Iw0 - wb0*Iw2;
    float wIw2 = wb0*Iw1 - wb1*Iw0;

    float wd0 = (tb0 - wIw0) / Ixx;
    float wd1 = (tb1 - wIw1) / Iyy;
    float wd2 = (tb2 - wIw2) / Izz;

    float wn0 = wb0 + wd0*dt;
    float wn1 = wb1 + wd1*dt;
    float wn2 = wb2 + wd2*dt;

    w_new[b*3 + 0] = wn0;
    w_new[b*3 + 1] = wn1;
    w_new[b*3 + 2] = wn2;

    // ---- Quaternion kinematics: qdot = 0.5 * Omega(w_new) @ q (wxyz) ----
    float dqw = 0.5f*(-wn0*qx - wn1*qy - wn2*qz);
    float dqx = 0.5f*( wn0*qw + wn2*qy - wn1*qz);
    float dqy = 0.5f*( wn1*qw - wn2*qx + wn0*qz);
    float dqz = 0.5f*( wn2*qw + wn1*qx - wn0*qy);

    float qnw = qw + dqw*dt;
    float qnx = qx + dqx*dt;
    float qny = qy + dqy*dt;
    float qnz = qz + dqz*dt;

    float norm_inv = rsqrtf(qnw*qnw + qnx*qnx + qny*qny + qnz*qnz + 1e-9f);
    q_new[b*4 + 0] = qnw * norm_inv;
    q_new[b*4 + 1] = qnx * norm_inv;
    q_new[b*4 + 2] = qny * norm_inv;
    q_new[b*4 + 3] = qnz * norm_inv;

    // ---- Position and velocity update ----
    p_new[b*3 + 0] = px + v[b*3 + 0]*dt;
    p_new[b*3 + 1] = py + v[b*3 + 1]*dt;
    p_new[b*3 + 2] = pz + v[b*3 + 2]*dt;

    v_new[b*3 + 0] = v[b*3 + 0] + ax*dt;
    v_new[b*3 + 1] = v[b*3 + 1] + ay*dt;
    v_new[b*3 + 2] = v[b*3 + 2] + az*dt;
}

// ---------------------------------------------------------------------------
// C++ launcher functions — called from srbd_ext.cpp
// ---------------------------------------------------------------------------

std::vector<torch::Tensor> srbd_step_cuda_launch(
    torch::Tensor p,
    torch::Tensor v,
    torch::Tensor q,
    torch::Tensor w,
    torch::Tensor f_world,
    torch::Tensor q_ref12,
    float m, float g, float Ixx, float Iyy, float Izz, float dt
) {
    int B = static_cast<int>(p.size(0));

    auto opts = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(p.device());

    auto p_out = torch::empty({B, 3}, opts);
    auto v_out = torch::empty({B, 3}, opts);
    auto q_out = torch::empty({B, 4}, opts);
    auto w_out = torch::empty({B, 3}, opts);

    const int THREADS = 256;
    const int BLOCKS = (B + THREADS - 1) / THREADS;
    srbd_step_kernel<<<BLOCKS, THREADS>>>(
        p.data_ptr<float>(),
        v.data_ptr<float>(),
        q.data_ptr<float>(),
        w.data_ptr<float>(),
        f_world.data_ptr<float>(),
        q_ref12.data_ptr<float>(),
        p_out.data_ptr<float>(),
        v_out.data_ptr<float>(),
        q_out.data_ptr<float>(),
        w_out.data_ptr<float>(),
        m, g, Ixx, Iyy, Izz, dt,
        B
    );

    return {p_out, v_out, q_out, w_out};
}

torch::Tensor foot_positions_cuda_launch(
    torch::Tensor p_base,
    torch::Tensor q,
    torch::Tensor q_ref12
) {
    int B = static_cast<int>(p_base.size(0));

    auto p_foot = torch::empty(
        {B, 4, 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(p_base.device())
    );

    const int THREADS = 256;
    const int BLOCKS = (B + THREADS - 1) / THREADS;
    foot_positions_kernel<<<BLOCKS, THREADS>>>(
        p_base.data_ptr<float>(),
        q.data_ptr<float>(),
        q_ref12.data_ptr<float>(),
        p_foot.data_ptr<float>(),
        B
    );

    return p_foot;
}

// ---------------------------------------------------------------------------
// Kernel 3: srbd_step_backward_kernel
//
// Hand-written analytic adjoint of srbd_step_kernel.
//
// Grid : ((B+255)/256,)   Block: (256,)   — one thread per environment
//
// Phase 1 recomputes all forward intermediates (R, per-foot leg_local/r,
//   fsum, tau_world, tau_body, Iw, wxIw, wdot, wn, qn, n_inv, q_new) using
//   the exact same arithmetic as the forward, so gradient computations see
//   bit-identical inputs.
// Phase 2 applies the chain rule in reverse order: translational adjoints,
//   normalize, qn = q + dq*dt, Omega(wn)@q, wn = w + wdot*dt, wdot,
//   wxIw = w×(Iw), tau_body = R^T@tau_world, per-foot cross product and
//   r = R@leg_local, leg_local → ox/oz → q_ref12, finally R → q.
// All accumulators live in registers; small per-foot arrays (leg_local, r)
//   are kept in registers via #pragma unroll on the f=0..3 loops.
// ---------------------------------------------------------------------------
__global__ void srbd_step_backward_kernel(
    // forward inputs (re-read for recompute)
    const float* __restrict__ p,         // [B, 3]
    const float* __restrict__ v,         // [B, 3]   (only dt-weighted into g_v)
    const float* __restrict__ q,         // [B, 4]
    const float* __restrict__ w,         // [B, 3]
    const float* __restrict__ f_world,   // [B, 4, 3]
    const float* __restrict__ q_ref12,   // [B, 12]
    // upstream gradients
    const float* __restrict__ g_p_new,   // [B, 3]
    const float* __restrict__ g_v_new,   // [B, 3]
    const float* __restrict__ g_q_new,   // [B, 4]
    const float* __restrict__ g_w_new,   // [B, 3]
    // outputs
    float*       __restrict__ g_p,        // [B, 3]
    float*       __restrict__ g_v,        // [B, 3]
    float*       __restrict__ g_q,        // [B, 4]
    float*       __restrict__ g_w,        // [B, 3]
    float*       __restrict__ g_f_world,  // [B, 4, 3]
    float*       __restrict__ g_q_ref12,  // [B, 12]
    float m, float g_grav,
    float Ixx, float Iyy, float Izz,
    float dt,
    int B
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    // ====================================================================
    // PHASE 1: Recompute forward intermediates (must match srbd_step_kernel
    // arithmetic exactly so gradients use the same intermediates the forward
    // produced).
    // ====================================================================
    float qw = q[b*4 + 0];
    float qx = q[b*4 + 1];
    float qy = q[b*4 + 2];
    float qz = q[b*4 + 3];
    float R[9];
    quat_to_R(qw, qx, qy, qz, R);

    float w0 = w[b*3 + 0];
    float w1 = w[b*3 + 1];
    float w2 = w[b*3 + 2];

    // Per-foot intermediates kept for backward (constant indices + unroll
    // keep these in registers).
    float leg_local[4][3];
    float r_arr[4][3];

    float fsum_x = 0.0f;
    float fsum_y = 0.0f;
    float fsum_z = -m * g_grav;
    float tau_x = 0.0f;
    float tau_y = 0.0f;
    float tau_z = 0.0f;

    #pragma unroll
    for (int f = 0; f < 4; ++f) {
        float q2 = q_ref12[b*12 + f*3 + 1];
        float q3 = q_ref12[b*12 + f*3 + 2];
        float ox = L1*sinf(q2) + L2*sinf(q2 + q3);
        float oz = -L1*cosf(q2) - L2*cosf(q2 + q3);

        // leg_local[f] = hip_offsets[f] + (ox, 0, oz)
        float ll0 = c_hip_offsets[f][0] + ox;
        float ll1 = c_hip_offsets[f][1];
        float ll2 = oz;
        leg_local[f][0] = ll0;
        leg_local[f][1] = ll1;
        leg_local[f][2] = ll2;

        // r[f] = R @ leg_local[f]  (p_foot - p_com; p cancels analytically)
        float rx = R[0]*ll0 + R[1]*ll1 + R[2]*ll2;
        float ry = R[3]*ll0 + R[4]*ll1 + R[5]*ll2;
        float rz = R[6]*ll0 + R[7]*ll1 + R[8]*ll2;
        r_arr[f][0] = rx;
        r_arr[f][1] = ry;
        r_arr[f][2] = rz;

        float fx = f_world[b*12 + f*3 + 0];
        float fy = f_world[b*12 + f*3 + 1];
        float fz = f_world[b*12 + f*3 + 2];

        fsum_x += fx;
        fsum_y += fy;
        fsum_z += fz;

        tau_x += ry*fz - rz*fy;
        tau_y += rz*fx - rx*fz;
        tau_z += rx*fy - ry*fx;
    }

    // tau_body = R^T @ tau_world
    float tb0 = R[0]*tau_x + R[3]*tau_y + R[6]*tau_z;
    float tb1 = R[1]*tau_x + R[4]*tau_y + R[7]*tau_z;
    float tb2 = R[2]*tau_x + R[5]*tau_y + R[8]*tau_z;

    // Euler's equation
    float Iw0 = Ixx * w0;
    float Iw1 = Iyy * w1;
    float Iw2 = Izz * w2;
    float wIw0 = w1*Iw2 - w2*Iw1;
    float wIw1 = w2*Iw0 - w0*Iw2;
    float wIw2 = w0*Iw1 - w1*Iw0;
    float wd0 = (tb0 - wIw0) / Ixx;
    float wd1 = (tb1 - wIw1) / Iyy;
    float wd2 = (tb2 - wIw2) / Izz;
    float wn0 = w0 + wd0*dt;
    float wn1 = w1 + wd1*dt;
    float wn2 = w2 + wd2*dt;

    // Quaternion kinematics (uses wn)
    float dqw_v = 0.5f*(-wn0*qx - wn1*qy - wn2*qz);
    float dqx_v = 0.5f*( wn0*qw + wn2*qy - wn1*qz);
    float dqy_v = 0.5f*( wn1*qw - wn2*qx + wn0*qz);
    float dqz_v = 0.5f*( wn2*qw + wn1*qx - wn0*qy);
    float qnw = qw + dqw_v*dt;
    float qnx = qx + dqx_v*dt;
    float qny = qy + dqy_v*dt;
    float qnz = qz + dqz_v*dt;
    float n_inv = rsqrtf(qnw*qnw + qnx*qnx + qny*qny + qnz*qnz + 1e-9f);
    float q_new_w = qnw * n_inv;
    float q_new_x = qnx * n_inv;
    float q_new_y = qny * n_inv;
    float q_new_z = qnz * n_inv;

    // ====================================================================
    // PHASE 2: Backward (analytical chain rule, top-down).
    // ====================================================================
    float g_p_new_x = g_p_new[b*3 + 0];
    float g_p_new_y = g_p_new[b*3 + 1];
    float g_p_new_z = g_p_new[b*3 + 2];
    float g_v_new_x = g_v_new[b*3 + 0];
    float g_v_new_y = g_v_new[b*3 + 1];
    float g_v_new_z = g_v_new[b*3 + 2];
    float g_q_new_w = g_q_new[b*4 + 0];
    float g_q_new_x = g_q_new[b*4 + 1];
    float g_q_new_y = g_q_new[b*4 + 2];
    float g_q_new_z = g_q_new[b*4 + 3];
    float g_w_new_0 = g_w_new[b*3 + 0];
    float g_w_new_1 = g_w_new[b*3 + 1];
    float g_w_new_2 = g_w_new[b*3 + 2];

    // -- B1: p_new = p + v*dt;  v_new = v + a*dt, a = fsum/m --
    float g_p_x = g_p_new_x;
    float g_p_y = g_p_new_y;
    float g_p_z = g_p_new_z;
    float g_v_x = g_p_new_x * dt + g_v_new_x;
    float g_v_y = g_p_new_y * dt + g_v_new_y;
    float g_v_z = g_p_new_z * dt + g_v_new_z;
    float g_fsum_x = g_v_new_x * (dt / m);
    float g_fsum_y = g_v_new_y * (dt / m);
    float g_fsum_z = g_v_new_z * (dt / m);

    // -- B2: q_new = qn * n_inv  (normalize) --
    float qn_dot_g = q_new_w*g_q_new_w + q_new_x*g_q_new_x
                   + q_new_y*g_q_new_y + q_new_z*g_q_new_z;
    float g_qn_w = n_inv * (g_q_new_w - q_new_w * qn_dot_g);
    float g_qn_x = n_inv * (g_q_new_x - q_new_x * qn_dot_g);
    float g_qn_y = n_inv * (g_q_new_y - q_new_y * qn_dot_g);
    float g_qn_z = n_inv * (g_q_new_z - q_new_z * qn_dot_g);

    // -- B3: qn = q + dq*dt  (g_q gets more from B4 and B11 below) --
    float g_q_w = g_qn_w;
    float g_q_x = g_qn_x;
    float g_q_y = g_qn_y;
    float g_q_z = g_qn_z;
    float g_dq_w = g_qn_w * dt;
    float g_dq_x = g_qn_x * dt;
    float g_dq_y = g_qn_y * dt;
    float g_dq_z = g_qn_z * dt;

    // -- B4: dq = 0.5 * Omega(wn) @ q  →  grads into q and wn --
    g_q_w += 0.5f * ( wn0*g_dq_x + wn1*g_dq_y + wn2*g_dq_z);
    g_q_x += 0.5f * (-wn0*g_dq_w - wn2*g_dq_y + wn1*g_dq_z);
    g_q_y += 0.5f * (-wn1*g_dq_w + wn2*g_dq_x - wn0*g_dq_z);
    g_q_z += 0.5f * (-wn2*g_dq_w - wn1*g_dq_x + wn0*g_dq_y);

    float g_wn_from_dq_0 = 0.5f * (-qx*g_dq_w + qw*g_dq_x + qz*g_dq_y - qy*g_dq_z);
    float g_wn_from_dq_1 = 0.5f * (-qy*g_dq_w - qz*g_dq_x + qw*g_dq_y + qx*g_dq_z);
    float g_wn_from_dq_2 = 0.5f * (-qz*g_dq_w + qy*g_dq_x - qx*g_dq_y + qw*g_dq_z);

    // -- B5: combine, propagate through wn = w + wdot*dt --
    float g_wn_0 = g_w_new_0 + g_wn_from_dq_0;
    float g_wn_1 = g_w_new_1 + g_wn_from_dq_1;
    float g_wn_2 = g_w_new_2 + g_wn_from_dq_2;
    float g_w_0 = g_wn_0;
    float g_w_1 = g_wn_1;
    float g_w_2 = g_wn_2;
    float g_wdot_0 = g_wn_0 * dt;
    float g_wdot_1 = g_wn_1 * dt;
    float g_wdot_2 = g_wn_2 * dt;

    // -- B6: wdot = (tau_body - wxIw) / I_diag --
    float g_tb_0 = g_wdot_0 / Ixx;
    float g_tb_1 = g_wdot_1 / Iyy;
    float g_tb_2 = g_wdot_2 / Izz;
    float g_wxIw_0 = -g_tb_0;
    float g_wxIw_1 = -g_tb_1;
    float g_wxIw_2 = -g_tb_2;

    // -- B7: wxIw = w × Iw (with Iw = I_diag * w; expanded form) --
    g_w_0 += g_wxIw_1 * w2 * (Ixx - Izz) + g_wxIw_2 * w1 * (Iyy - Ixx);
    g_w_1 += g_wxIw_0 * w2 * (Izz - Iyy) + g_wxIw_2 * w0 * (Iyy - Ixx);
    g_w_2 += g_wxIw_0 * w1 * (Izz - Iyy) + g_wxIw_1 * w0 * (Ixx - Izz);

    // -- B8: tau_body = R^T @ tau_world --
    // g_tau_world[j] = (R @ g_tau_body)[j]
    float g_tw_0 = R[0]*g_tb_0 + R[1]*g_tb_1 + R[2]*g_tb_2;
    float g_tw_1 = R[3]*g_tb_0 + R[4]*g_tb_1 + R[5]*g_tb_2;
    float g_tw_2 = R[6]*g_tb_0 + R[7]*g_tb_1 + R[8]*g_tb_2;
    // g_R[j*3+i] += g_tau_body[i] * tau_world[j]   (outer product)
    float g_R[9];
    g_R[0] = g_tb_0 * tau_x;
    g_R[1] = g_tb_1 * tau_x;
    g_R[2] = g_tb_2 * tau_x;
    g_R[3] = g_tb_0 * tau_y;
    g_R[4] = g_tb_1 * tau_y;
    g_R[5] = g_tb_2 * tau_y;
    g_R[6] = g_tb_0 * tau_z;
    g_R[7] = g_tb_1 * tau_z;
    g_R[8] = g_tb_2 * tau_z;

    // -- B9/B10: per-foot backward (cross product, r = R@leg_local, FK chain)
    // Sign convention for c = a × b: grad_a = b × grad_c; grad_b = grad_c × a.
    #pragma unroll
    for (int f = 0; f < 4; ++f) {
        float rx = r_arr[f][0];
        float ry = r_arr[f][1];
        float rz = r_arr[f][2];
        float fx = f_world[b*12 + f*3 + 0];
        float fy = f_world[b*12 + f*3 + 1];
        float fz = f_world[b*12 + f*3 + 2];

        // g_r = f × g_tw
        float g_r_x = fy*g_tw_2 - fz*g_tw_1;
        float g_r_y = fz*g_tw_0 - fx*g_tw_2;
        float g_r_z = fx*g_tw_1 - fy*g_tw_0;

        // g_f_world = g_tw × r  + B1's g_fsum
        g_f_world[b*12 + f*3 + 0] = g_tw_1*rz - g_tw_2*ry + g_fsum_x;
        g_f_world[b*12 + f*3 + 1] = g_tw_2*rx - g_tw_0*rz + g_fsum_y;
        g_f_world[b*12 + f*3 + 2] = g_tw_0*ry - g_tw_1*rx + g_fsum_z;

        // r = R @ leg_local:  g_R[c*3+k] += g_r[c] * leg_local[k]
        float ll0 = leg_local[f][0];
        float ll1 = leg_local[f][1];
        float ll2 = leg_local[f][2];
        g_R[0] += g_r_x * ll0;
        g_R[1] += g_r_x * ll1;
        g_R[2] += g_r_x * ll2;
        g_R[3] += g_r_y * ll0;
        g_R[4] += g_r_y * ll1;
        g_R[5] += g_r_y * ll2;
        g_R[6] += g_r_z * ll0;
        g_R[7] += g_r_z * ll1;
        g_R[8] += g_r_z * ll2;

        // g_leg_local = R^T @ g_r;  only the [0] and [2] entries matter
        // (leg_local[1] = hip_offsets[f,1] is constant → no grad).
        float g_ll0 = R[0]*g_r_x + R[3]*g_r_y + R[6]*g_r_z;
        float g_ll2 = R[2]*g_r_x + R[5]*g_r_y + R[8]*g_r_z;
        float g_ox = g_ll0;
        float g_oz = g_ll2;

        // ox = L1 sin(q2) + L2 sin(q2+q3); oz = -L1 cos(q2) - L2 cos(q2+q3)
        float q2 = q_ref12[b*12 + f*3 + 1];
        float q3 = q_ref12[b*12 + f*3 + 2];
        float cos_q2  = cosf(q2);
        float sin_q2  = sinf(q2);
        float cos_q23 = cosf(q2 + q3);
        float sin_q23 = sinf(q2 + q3);

        g_q_ref12[b*12 + f*3 + 0] = 0.0f;
        g_q_ref12[b*12 + f*3 + 1] = g_ox * (L1*cos_q2 + L2*cos_q23)
                                  + g_oz * (L1*sin_q2 + L2*sin_q23);
        g_q_ref12[b*12 + f*3 + 2] = g_ox * (L2*cos_q23)
                                  + g_oz * (L2*sin_q23);
    }

    // -- B11: R = quat_to_R(q)  →  g_q from grouped formula --
    g_q_w += 2.0f * ( qx*(g_R[7] - g_R[5])
                    + qy*(g_R[2] - g_R[6])
                    + qz*(g_R[3] - g_R[1]) );
    g_q_x += 2.0f * ( qy*(g_R[1] + g_R[3])
                    + qz*(g_R[2] + g_R[6])
                    + qw*(g_R[7] - g_R[5]) )
           - 4.0f * qx*(g_R[4] + g_R[8]);
    g_q_y += 2.0f * ( qx*(g_R[1] + g_R[3])
                    + qz*(g_R[5] + g_R[7])
                    + qw*(g_R[2] - g_R[6]) )
           - 4.0f * qy*(g_R[0] + g_R[8]);
    g_q_z += 2.0f * ( qx*(g_R[2] + g_R[6])
                    + qy*(g_R[5] + g_R[7])
                    + qw*(g_R[3] - g_R[1]) )
           - 4.0f * qz*(g_R[0] + g_R[4]);

    // ====================================================================
    // PHASE 3: Write outputs.
    // ====================================================================
    g_p[b*3 + 0] = g_p_x;
    g_p[b*3 + 1] = g_p_y;
    g_p[b*3 + 2] = g_p_z;
    g_v[b*3 + 0] = g_v_x;
    g_v[b*3 + 1] = g_v_y;
    g_v[b*3 + 2] = g_v_z;
    g_q[b*4 + 0] = g_q_w;
    g_q[b*4 + 1] = g_q_x;
    g_q[b*4 + 2] = g_q_y;
    g_q[b*4 + 3] = g_q_z;
    g_w[b*3 + 0] = g_w_0;
    g_w[b*3 + 1] = g_w_1;
    g_w[b*3 + 2] = g_w_2;
}

std::vector<torch::Tensor> srbd_step_backward_cuda_launch(
    torch::Tensor p,
    torch::Tensor v,
    torch::Tensor q,
    torch::Tensor w,
    torch::Tensor f_world,
    torch::Tensor q_ref12,
    torch::Tensor g_p_new,
    torch::Tensor g_v_new,
    torch::Tensor g_q_new,
    torch::Tensor g_w_new,
    float m, float g_grav, float Ixx, float Iyy, float Izz, float dt
) {
    int B = static_cast<int>(p.size(0));

    auto g_p        = torch::empty_like(p);
    auto g_v        = torch::empty_like(v);
    auto g_q        = torch::empty_like(q);
    auto g_w        = torch::empty_like(w);
    auto g_f_world  = torch::empty_like(f_world);
    auto g_q_ref12  = torch::empty_like(q_ref12);

    const int THREADS = 256;
    const int BLOCKS = (B + THREADS - 1) / THREADS;
    srbd_step_backward_kernel<<<BLOCKS, THREADS>>>(
        p.data_ptr<float>(),
        v.data_ptr<float>(),
        q.data_ptr<float>(),
        w.data_ptr<float>(),
        f_world.data_ptr<float>(),
        q_ref12.data_ptr<float>(),
        g_p_new.data_ptr<float>(),
        g_v_new.data_ptr<float>(),
        g_q_new.data_ptr<float>(),
        g_w_new.data_ptr<float>(),
        g_p.data_ptr<float>(),
        g_v.data_ptr<float>(),
        g_q.data_ptr<float>(),
        g_w.data_ptr<float>(),
        g_f_world.data_ptr<float>(),
        g_q_ref12.data_ptr<float>(),
        m, g_grav, Ixx, Iyy, Izz, dt,
        B
    );

    return {g_p, g_v, g_q, g_w, g_f_world, g_q_ref12};
}
