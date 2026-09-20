"""Rectified flow: linear-interpolation flow matching.

Conventions (kept consistent with the report and the samplers):

    t = 0 -> data,   t = 1 -> noise
    x_t        = (1 - t) * x0 + t * noise
    v*(x_t, t) = noise - x0

Sampling integrates the ODE backwards, from t = 1 (noise) to t = 0 (data):

    x_{t-dt} = x_t - dt * v_theta(x_t, t)

The model interface mirrors DDPM (`model(x, t)` returns the same shape as `x`)
so a single training loop can drive either objective.
"""

import torch
import torch.nn.functional as F

# The sinusoidal timestep embedding is calibrated for a range of ~[0, 1000]
# (as in DDPM). The path math stays in t in [0, 1]; the model sees t * T_SCALE.
T_SCALE = 1000.0


class RectifiedFlow:
    def __init__(self, model, device="cpu", img_channels=1, img_size=28,
                 t_sample="uniform", logit_normal_std=1.0, t_scale=T_SCALE):
        self.model = model
        self.device = device
        self.img_channels = img_channels
        self.img_size = img_size
        self.t_sample = t_sample
        self.logit_normal_std = logit_normal_std
        self.t_scale = t_scale

    def sample_t(self, n):
        if self.t_sample == "logit_normal":
            u = torch.randn(n, device=self.device) * self.logit_normal_std
            return torch.sigmoid(u)
        return torch.rand(n, device=self.device)

    def forward_path(self, x0, t):
        """Returns (x_t, v_target) along the straight noise->data path."""
        noise = torch.randn_like(x0)
        t_ = t.view(-1, 1, 1, 1)
        x_t = (1.0 - t_) * x0 + t_ * noise
        v_target = noise - x0
        return x_t, v_target

    def training_loss(self, x0):
        t = self.sample_t(x0.shape[0])
        x_t, v_target = self.forward_path(x0, t)
        v_pred = self.model(x_t, t * self.t_scale)
        return F.mse_loss(v_pred, v_target)

    @torch.no_grad()
    def sample(self, n, num_steps=32, return_all=False):
        self.model.eval()
        x = torch.randn(n, self.img_channels, self.img_size, self.img_size,
                        device=self.device)
        trajectory = [x.clone()] if return_all else None
        capture_every = max(1, num_steps // 10)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = 1.0 - i * dt
            t_batch = torch.full((n,), t, device=self.device)
            v = self.model(x, t_batch * self.t_scale)
            x = x - dt * v
            if return_all and (i + 1) % capture_every == 0:
                trajectory.append(x.clone())

        self.model.train()
        x = torch.clamp(x, -1.0, 1.0)
        if return_all:
            return x, trajectory
        return x
