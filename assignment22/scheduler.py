"""
Noise Scheduler for DDPM.

Implements linear and cosine beta schedules, and pre-computes every
derived quantity (alpha, alpha_bar / alphas_cumprod, and the various
square-roots) needed by the forward process and the sampler.

Reference: Ho et al. 2020 (DDPM) for the linear schedule,
           Nichol & Dhariwal 2021 (Improved DDPM) for the cosine schedule.
"""
import math
import torch


class NoiseScheduler:
    def __init__(self, timesteps=1000, schedule="linear",
                 beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.T = timesteps
        self.device = device
        self.schedule = schedule

        if schedule == "linear":
            betas = self._linear_beta_schedule(timesteps, beta_start, beta_end)
        elif schedule == "cosine":
            betas = self._cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown schedule '{schedule}', use 'linear' or 'cosine'")

        self.betas = betas.to(device)                                   # beta_t
        self.alphas = 1.0 - self.betas                                  # alpha_t
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)         # alpha_bar_t
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]]
        )

        # Quantities used by q_sample (forward process)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Quantities used by the reverse (sampling) process
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    @staticmethod
    def _linear_beta_schedule(timesteps, beta_start, beta_end):
        return torch.linspace(beta_start, beta_end, timesteps)

    @staticmethod
    def _cosine_beta_schedule(timesteps, s=0.008):
        """Nichol & Dhariwal cosine schedule, clipped to keep betas well behaved."""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 1e-4, 0.9999)

    def extract(self, a, t, x_shape):
        """
        Gather values from a 1-D tensor `a` (length T) at indices `t` (shape (B,))
        and reshape to (B, 1, 1, ...) so it broadcasts against an image batch
        of shape x_shape = (B, C, H, W).
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def __repr__(self):
        return f"NoiseScheduler(T={self.T}, schedule='{self.schedule}')"
