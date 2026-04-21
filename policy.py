# ⚠️ CRITICAL: Isaac Gym must be imported BEFORE torch
try:
    from isaacgym import gymapi
except Exception:
    pass

import torch
import torch.nn as nn


class Policy(nn.Module):
    """Neural network policy: input 36-dim observation -> 256 * 256 -> output 12-dim joint angle offset"""

    def __init__(self, dim_obs=36, dim_action=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_obs, 256), nn.LeakyReLU(0.05),
            nn.Linear(256, 256), nn.LeakyReLU(0.05),
            nn.Linear(256, dim_action),
        )
        with torch.no_grad():
            nn.init.normal_(self.net[-1].weight, std=1e-2)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, s, h=None):
        return self.net(s), None

    def reset(self):
        pass
