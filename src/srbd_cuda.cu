/*
 * srbd_cuda.cu — Custom CUDA kernels for the SRBD (Single Rigid Body Dynamics) model.
 *
 * Two kernels are provided:
 *   1. foot_positions_kernel  — computes world-frame foot positions from SRBD state + joint angles
 *   2. srbd_step_kernel       — fused kernel: foot FK + Newton-Euler dynamics + quaternion integration
 *
 * Thread layout:
 *   foot_positions_kernel : grid=(B,)  block=(4,3)   one thread per (foot, xyz) pair
 *   srbd_step_kernel      : grid=(B,)  block=(32,)   one warp per environment
 *
 * All tensors are float32, stored in row-major (C) order.
 * Requires CUDA compute capability >= 6.0 (Pascal) for shared-memory float atomicAdd.
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
// Grid : (B,)       — one block per environment
// Block: (4, 3)     — one thread per (foot_index, xyz_component)
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
    int b = blockIdx.x;   // environment index
    int f = threadIdx.x;  // foot index  [0, 3]
    int c = threadIdx.y;  // xyz component [0, 2]

    if (b >= B) return;

    // Thread (0,0) builds the rotation matrix and stores it in shared memory.
    // All other threads wait at __syncthreads() before reading it.
    __shared__ float sh_R[9];
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        float qw = q[b*4 + 0];
        float qx = q[b*4 + 1];
        float qy = q[b*4 + 2];
        float qz = q[b*4 + 3];
        quat_to_R(qw, qx, qy, qz, sh_R);
    }
    __syncthreads();

    // Each thread computes one scalar element of the (4,3) output array.
    // p_foot[b, f, c] = hip_world[c] + off_world[c]

    // --- Hip position in world frame ---
    // hip_world[c] = p_base[b, c]  +  R[c, :] · hip_offsets[f, :]
    float hip_w = p_base[b*3 + c];
    for (int k = 0; k < 3; ++k) {
        hip_w += sh_R[c*3 + k] * c_hip_offsets[f][k];
    }

    // --- Leg forward kinematics (2-link planar in the sagittal plane) ---
    // q_ref12 layout: [hip, thigh, calf] repeated for FL, FR, RL, RR
    float q2 = q_ref12[b*12 + f*3 + 1];  // thigh joint angle
    float q3 = q_ref12[b*12 + f*3 + 2];  // calf  joint angle

    // Foot offset in body frame (y-component is always 0 for this 2-link model)
    float ox = L1*sinf(q2) + L2*sinf(q2 + q3);
    float oz = -L1*cosf(q2) - L2*cosf(q2 + q3);
    // oy = 0, so off_world[c] = R[c,0]*ox + R[c,2]*oz

    float off_w = sh_R[c*3 + 0]*ox + sh_R[c*3 + 2]*oz;

    p_foot[b*12 + f*3 + c] = hip_w + off_w;
}

// ---------------------------------------------------------------------------
// Kernel 2: srbd_step_kernel
//
// Fused kernel: foot FK + centroidal dynamics + quaternion integration.
// Implements one Euler integration step of the SRBD equations of motion.
//
// Grid : (B,)    — one block per environment
// Block: (32,)   — one warp per environment (minimal sync overhead)
//
// Thread roles:
//   Threads  0-11 : compute foot positions (foot 0-3, xyz 0-2)
//                   then accumulate force sum and world torque
//   Thread   0    : finishes translational accel, Euler eq, quat integration,
//                   and writes all four output state vectors
//   Threads 12-31 : idle after foot/force step
//
// Inputs  (row-major):
//   p       [B, 3]    srbd_p  — body CoM position (world)
//   v       [B, 3]    srbd_v  — body CoM velocity (world)
//   q       [B, 4]    srbd_q  — body orientation quaternion (wxyz)
//   w       [B, 3]    srbd_w  — body angular velocity (body frame)
//   f_world [B, 4, 3] — ground reaction forces per foot (world frame)
//   q_ref12 [B, 12]  — reference joint angles (for foot FK)
//   m, g             — total mass and gravity magnitude
//   Ixx, Iyy, Izz   — principal inertia values (diagonal I matrix, body frame)
//   dt               — integration timestep
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
    int b   = blockIdx.x;   // environment index
    int tid = threadIdx.x;  // thread index in warp [0, 31]

    if (b >= B) return;

    // Shared memory layout (total ~156 bytes per block):
    //   sh_R[9]       — rotation matrix R (body->world) for this env
    //   sh_p_foot[12] — foot world positions (4 feet × 3 coords)
    //   sh_f_sum[3]   — accumulated total external force (includes gravity)
    //   sh_tau_w[3]   — accumulated world-frame torque
    __shared__ float sh_R[9];
    __shared__ float sh_p_foot[12];
    __shared__ float sh_f_sum[3];
    __shared__ float sh_tau_w[3];

    // -----------------------------------------------------------------------
    // Step 1: Thread 0 computes R and initialises accumulators
    // -----------------------------------------------------------------------
    if (tid == 0) {
        float qw = q[b*4 + 0];
        float qx = q[b*4 + 1];
        float qy = q[b*4 + 2];
        float qz = q[b*4 + 3];
        quat_to_R(qw, qx, qy, qz, sh_R);

        // Pre-load gravity into force accumulator (gravity acts on CoM, downward)
        sh_f_sum[0] = 0.0f;
        sh_f_sum[1] = 0.0f;
        sh_f_sum[2] = -m * g;

        sh_tau_w[0] = 0.0f;
        sh_tau_w[1] = 0.0f;
        sh_tau_w[2] = 0.0f;
    }
    __syncthreads();

    // -----------------------------------------------------------------------
    // Step 2: Threads 0-11 compute foot positions (one thread per foot-component)
    // -----------------------------------------------------------------------
    if (tid < 12) {
        int f = tid / 3;  // foot index
        int c = tid % 3;  // xyz component

        // Hip in world frame
        float hip_w = p[b*3 + c];
        for (int k = 0; k < 3; ++k) {
            hip_w += sh_R[c*3 + k] * c_hip_offsets[f][k];
        }

        // 2-link leg FK
        float q2 = q_ref12[b*12 + f*3 + 1];
        float q3 = q_ref12[b*12 + f*3 + 2];
        float ox = L1*sinf(q2) + L2*sinf(q2 + q3);
        float oz = -L1*cosf(q2) - L2*cosf(q2 + q3);
        float off_w = sh_R[c*3 + 0]*ox + sh_R[c*3 + 2]*oz;

        sh_p_foot[f*3 + c] = hip_w + off_w;
    }
    __syncthreads();

    // -----------------------------------------------------------------------
    // Step 3: Threads 0-11 accumulate force sum and world torque
    //
    // tau_world = sum_f  cross(r_f, f_f)  where r_f = p_foot_f - p_com
    // We split the cross product computation across components:
    //   tid % 3 == 0 -> computes tau_x contribution for foot (tid/3)
    //   tid % 3 == 1 -> computes tau_y contribution for foot (tid/3)
    //   tid % 3 == 2 -> computes tau_z contribution for foot (tid/3)
    // -----------------------------------------------------------------------
    if (tid < 12) {
        int f = tid / 3;
        int c = tid % 3;

        // Accumulate force component c from foot f
        atomicAdd(&sh_f_sum[c], f_world[b*12 + f*3 + c]);

        // Moment arm: r = p_foot - p_com
        float rx = sh_p_foot[f*3 + 0] - p[b*3 + 0];
        float ry = sh_p_foot[f*3 + 1] - p[b*3 + 1];
        float rz = sh_p_foot[f*3 + 2] - p[b*3 + 2];

        float fx = f_world[b*12 + f*3 + 0];
        float fy = f_world[b*12 + f*3 + 1];
        float fz = f_world[b*12 + f*3 + 2];

        // cross(r, f) = [ry*fz - rz*fy,  rz*fx - rx*fz,  rx*fy - ry*fx]
        if (c == 0) atomicAdd(&sh_tau_w[0], ry*fz - rz*fy);
        if (c == 1) atomicAdd(&sh_tau_w[1], rz*fx - rx*fz);
        if (c == 2) atomicAdd(&sh_tau_w[2], rx*fy - ry*fx);
    }
    __syncthreads();

    // -----------------------------------------------------------------------
    // Step 4: Thread 0 completes the dynamics and writes outputs
    //
    // Equations implemented (all in SI units, float32):
    //   a        = F_total / m
    //   tau_body = R^T @ tau_world
    //   wdot     = (tau_body - w × (I·w)) / I     [Euler's equation, body frame]
    //   w_new    = w + wdot * dt
    //   qdot     = 0.5 * Omega(w_new) @ q          [quaternion kinematics]
    //   q_new    = normalize(q + qdot * dt)
    //   p_new    = p + v * dt
    //   v_new    = v + a * dt
    // -----------------------------------------------------------------------
    if (tid == 0) {
        // --- Translational acceleration ---
        float ax = sh_f_sum[0] / m;
        float ay = sh_f_sum[1] / m;
        float az = sh_f_sum[2] / m;

        // --- tau_body = R^T @ tau_world ---
        // R^T has rows that are the columns of R, so R^T[i,j] = R[j*3+i]
        float tw0 = sh_tau_w[0];
        float tw1 = sh_tau_w[1];
        float tw2 = sh_tau_w[2];

        float tb0 = sh_R[0]*tw0 + sh_R[3]*tw1 + sh_R[6]*tw2;
        float tb1 = sh_R[1]*tw0 + sh_R[4]*tw1 + sh_R[7]*tw2;
        float tb2 = sh_R[2]*tw0 + sh_R[5]*tw1 + sh_R[8]*tw2;

        // --- Euler's equation: wdot = (tau_body - w × (I·w)) / I ---
        float wb0 = w[b*3 + 0];
        float wb1 = w[b*3 + 1];
        float wb2 = w[b*3 + 2];

        float Iw0 = Ixx * wb0;
        float Iw1 = Iyy * wb1;
        float Iw2 = Izz * wb2;

        // w × (I·w)
        float wIw0 = wb1*Iw2 - wb2*Iw1;
        float wIw1 = wb2*Iw0 - wb0*Iw2;
        float wIw2 = wb0*Iw1 - wb1*Iw0;

        float wd0 = (tb0 - wIw0) / Ixx;
        float wd1 = (tb1 - wIw1) / Iyy;
        float wd2 = (tb2 - wIw2) / Izz;

        // --- Angular velocity update ---
        float wn0 = wb0 + wd0*dt;
        float wn1 = wb1 + wd1*dt;
        float wn2 = wb2 + wd2*dt;

        w_new[b*3 + 0] = wn0;
        w_new[b*3 + 1] = wn1;
        w_new[b*3 + 2] = wn2;

        // --- Quaternion kinematics: qdot = 0.5 * Omega(w_new) @ q ---
        // Omega(w) is the 4x4 skew matrix for quaternion differentiation (wxyz convention):
        //   Omega = [[  0, -wx, -wy, -wz],
        //            [ wx,   0,  wz, -wy],
        //            [ wy, -wz,   0,  wx],
        //            [ wz,  wy, -wx,   0]]
        float qw = q[b*4 + 0];
        float qx = q[b*4 + 1];
        float qy = q[b*4 + 2];
        float qz = q[b*4 + 3];

        float dqw = 0.5f*(-wn0*qx - wn1*qy - wn2*qz);
        float dqx = 0.5f*( wn0*qw + wn2*qy - wn1*qz);
        float dqy = 0.5f*( wn1*qw - wn2*qx + wn0*qz);
        float dqz = 0.5f*( wn2*qw + wn1*qx - wn0*qy);

        float qnw = qw + dqw*dt;
        float qnx = qx + dqx*dt;
        float qny = qy + dqy*dt;
        float qnz = qz + dqz*dt;

        // Normalise (avoid division by near-zero)
        float norm_inv = rsqrtf(qnw*qnw + qnx*qnx + qny*qny + qnz*qnz + 1e-9f);
        q_new[b*4 + 0] = qnw * norm_inv;
        q_new[b*4 + 1] = qnx * norm_inv;
        q_new[b*4 + 2] = qny * norm_inv;
        q_new[b*4 + 3] = qnz * norm_inv;

        // --- Position and velocity update ---
        p_new[b*3 + 0] = p[b*3 + 0] + v[b*3 + 0]*dt;
        p_new[b*3 + 1] = p[b*3 + 1] + v[b*3 + 1]*dt;
        p_new[b*3 + 2] = p[b*3 + 2] + v[b*3 + 2]*dt;

        v_new[b*3 + 0] = v[b*3 + 0] + ax*dt;
        v_new[b*3 + 1] = v[b*3 + 1] + ay*dt;
        v_new[b*3 + 2] = v[b*3 + 2] + az*dt;
    }
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

    srbd_step_kernel<<<B, 32>>>(
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

    dim3 block(4, 3);
    foot_positions_kernel<<<B, block>>>(
        p_base.data_ptr<float>(),
        q.data_ptr<float>(),
        q_ref12.data_ptr<float>(),
        p_foot.data_ptr<float>(),
        B
    );

    return p_foot;
}
