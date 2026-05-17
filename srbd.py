# ⚠️ CRITICAL: Isaac Gym must be imported BEFORE torch
try:
    from isaacgym import gymapi
except Exception:
    pass

import torch

from utils_math import quat_rotate_inverse_wxyz, quat_to_rot


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

        # w_list = []
        # for b in range(self.B):
        #     w_b = quat_rotate_inverse_wxyz(self.srbd_q[b],
        #                                    self.base_ang_world[b],
        #                                    dev)
        #     w_list.append(w_b)
        # self.srbd_w = torch.stack(w_list, dim=0)       # (B,3)
        self.srbd_w = quat_rotate_inverse_wxyz(self.srbd_q, self.base_ang_world, dev) # (B,3)

    # Normalize quaternion to unit length

    def _quat_norm(self, q):
        if q.dim() == 1:
            return q / (q.norm() + 1e-9)
        else:
            return q / (q.norm(dim=-1, keepdim=True) + 1e-9)


    def _srbd_step(self, f_world, q_ref12, dt):
        """
        Full 3D centroidal dynamics step (batch, no inplace operations).
        f_world: (B,4,3)
        q_ref12: (B,12)
        """
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

        hip_offsets = torch.tensor([
            [ +0.1934, +0.1420, 0.0 ], # FL, FR, RL, RR
            [ +0.1934, -0.1420, 0.0 ],
            [ -0.1934, +0.1420, 0.0 ],
            [ -0.1934, -0.1420, 0.0 ],
        ], device=dev)                # (4,3)

        # First use “old state” to calculate foot positions and torques
        #p_foot = self.foot_positions_srbd(q_ref12.detach())  # (B,4,3)
        p_foot = self.foot_positions_srbd(q_ref12, hip_offsets)

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
        w_cross_Iw = torch.linalg.cross(w, Iw, dim=-1) # (B,3)
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


        # q_new_list = []
        # w_new_list = []

        # for b in range(self.B):
        #     # Arm length r: from COM to foot
        #     r = p_foot[b] - p[b].view(1, 3)        # (4,3)
        #     tau_world = torch.cross(r, f_world[b], dim=-1).sum(dim=0)  # (3,)

        #     # world -> body
        #     qw, qx, qy, qz = q[b]
        #     R = quat_to_rot(qw, qx, qy, qz, dev)   # (3,3)
        #     tau_body = R.t() @ tau_world           # (3,)

        #     # Euler equation I * wdot = tau - w × (I w)
        #     w_b = w[b]                             # (3,)

        #     Iw = I @ w_b                           # (3,)
        #     wdot = tau_body - torch.linalg.cross(w_b, Iw) / I # torch.linalg.solve(I, tau_body - torch.linalg.cross(w_b, Iw))        # (3,)



        #     w_b_new = w_b + wdot * dt              # (3,)

        #     wx, wy, wz = w_b_new.unbind()
        #     z = torch.zeros_like(wx)
        #     Omega = torch.stack([
        #         torch.stack([z,  -wx, -wy, -wz]),
        #         torch.stack([wx,  z,   wz, -wy]),
        #         torch.stack([wy, -wz,  z,   wx]),
        #         torch.stack([wz,  wy, -wx,  z ]),
        #     ], dim=0)
        #     qdot = 0.5 * (Omega @ q[b])
        #     q_b_new = self._quat_norm(q[b] + qdot * dt)

        #     w_new_list.append(w_b_new)
        #     q_new_list.append(q_b_new)

        # # ---- Actually update self.srbd_*, assign all at once (no inplace slicing) ----
        # p_new = p + v * dt                         # (B,3)
        # v_new = v + a * dt                         # (B,3)
        # q_new = torch.stack(q_new_list, dim=0)     # (B,4)
        # w_new = torch.stack(w_new_list, dim=0)     # (B,3)

        # self.srbd_p = p_new
        # self.srbd_v = v_new
        # self.srbd_q = q_new
        # self.srbd_w = w_new


    # ---------------- SRBD-based foot positions ----------------

    def foot_positions_srbd(self, q_ref12, hip_offsets):
        """
        3D SRBD foot positions (batch)
        q_ref12: (B,12)
        hip_offsets: (4,3)
        Returns (B,4,3)
        """
        dev = self.device
        B = self.B
        p_base = self.srbd_p          # (B,3)
        q = self.srbd_q               # (B,4)

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
        
        
        
        
        # out = torch.zeros(B, 4, 3, device=dev)

        # for b in range(B):
        #     qb = q[b]
        #     R = quat_to_rot(qb[0], qb[1], qb[2], qb[3], dev)

        #     hip_world = p_base[b].view(1,3) + (R @ hip_offsets.t()).t()  # (4,3)

        #     q_leg = q_ref12[b].view(4,3)
        #     q2, q3 = q_leg[:,1], q_leg[:,2]
        #     x = L1*torch.sin(q2) + L2*torch.sin(q2+q3)
        #     z = -L1*torch.cos(q2) - L2*torch.cos(q2+q3)
        #     y = torch.zeros_like(x)
        #     off_body = torch.stack([x,y,z], dim=1)  # (4,3)
        #     off_world = (R @ off_body.t()).t()
        #     out[b] = hip_world + off_world

        # return out  # (B,4,3)

    # ---------------- env step ----------------

