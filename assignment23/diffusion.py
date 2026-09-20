"""
Gaussian diffusion process (Assignment 2.3, self-contained folder).

Reused unchanged from Assignment 2:
  - q_sample(x0, t, noise)  : forward process
  - p_losses(x0, t, y)      : training loss (not used at inference time here,
                               kept only so this file matches the training
                               objective the checkpoint was optimized for)
  - p_sample / sample       : the ORIGINAL DDPM ancestral sampling loop
                               (kept only as a reference baseline to compare
                               against — this is the loop being replaced)

New in this assignment:
  - ddim_step   : ONE DDIM reverse update, x_t -> x_{t_prev}, split out as
                  its own method (rather than inlined in a loop) so it can
                  be unit-tested directly in verify_ddim.py against the
                  closed-form forward process, independent of the trained
                  network.
  - ddim_sample : the REPLACEMENT sampling loop — same idea as `sample`,
                  but walks a short, arbitrary subsequence of timesteps and
                  uses `ddim_step` instead of the DDPM posterior step.
"""
import torch
import torch.nn.functional as F


class GaussianDiffusion:
    def __init__(self, model, scheduler, device):
        self.model = model
        self.scheduler = scheduler
        self.device = device

    # ---------------------------------------------------------------- #
    # Forward process (reused from Assignment 2)
    # ---------------------------------------------------------------- #
    def q_sample(self, x0, t, noise=None):
        """x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise"""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ac = self.scheduler.extract(self.scheduler.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_omac = self.scheduler.extract(self.scheduler.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_ac * x0 + sqrt_omac * noise, noise

    def p_losses(self, x0, t, y=None, p_uncond=0.1):
        """Training objective (unused at inference; included for completeness)."""
        noise = torch.randn_like(x0)
        x_t, noise = self.q_sample(x0, t, noise)
        if y is not None and self.model.num_classes is not None and p_uncond > 0:
            drop_mask = torch.rand(y.shape[0], device=y.device) < p_uncond
            y = y.clone()
            y[drop_mask] = self.model.num_classes
        pred_noise = self.model(x_t, t, y)
        return F.mse_loss(pred_noise, noise)

    # ---------------------------------------------------------------- #
    # ORIGINAL sampling loop (DDPM, T=1000 steps) — reference baseline only
    # ---------------------------------------------------------------- #
    @torch.no_grad()
    def p_sample(self, x_t, t, t_index, y=None, guidance_scale=1.0):
        betas_t = self.scheduler.extract(self.scheduler.betas, t, x_t.shape)
        sqrt_omac_t = self.scheduler.extract(self.scheduler.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        sqrt_recip_alphas_t = self.scheduler.extract(self.scheduler.sqrt_recip_alphas, t, x_t.shape)

        pred_noise = self._predict_noise(x_t, t, y, guidance_scale)
        model_mean = sqrt_recip_alphas_t * (x_t - betas_t * pred_noise / sqrt_omac_t)

        if t_index == 0:
            return model_mean
        posterior_var_t = self.scheduler.extract(self.scheduler.posterior_variance, t, x_t.shape)
        noise = torch.randn_like(x_t)
        return model_mean + torch.sqrt(posterior_var_t) * noise

    @torch.no_grad()
    def sample(self, shape, y=None, guidance_scale=1.0):
        """Full T-step DDPM loop — this is the loop Assignment 2.3 replaces."""
        device = self.device
        x = torch.randn(shape, device=device)
        for i in reversed(range(self.scheduler.T)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t, i, y=y, guidance_scale=guidance_scale)
        return x

    # ---------------------------------------------------------------- #
    # REPLACEMENT sampling loop: DDIM  (this assignment's deliverable)
    # ---------------------------------------------------------------- #
    def ddim_step(self, x_t, eps, t, t_prev, eta):
        """
        One DDIM reverse update x_t -> x_{t_prev}, given the model's noise
        prediction `eps` at timestep `t` (Song et al., 2020, eq. 12).

        t, t_prev : plain Python ints (t_prev = -1 means "step to x_0").
        eta = 0   -> deterministic DDIM
        eta = 1   -> matches the DDPM posterior variance for adjacent steps
                     (see verify_ddim.py, Test 2)

        Pulled out as its own method (not inlined in a loop) specifically so
        it can be called directly, with a hand-crafted `eps`, in
        verify_ddim.py -- that test does not touch the neural network at
        all, so it isolates correctness of this formula from model quality.
        """
        alphas_cumprod = self.scheduler.alphas_cumprod
        a_t = alphas_cumprod[t]
        a_prev = alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=x_t.device)

        # 1) predict x0 from x_t and the noise estimate
        x0_pred = (x_t - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t)
        x0_pred = x0_pred.clamp(-1.0, 1.0)

        # 2) DDIM's interpolation back to a *lower*-noise x_{t_prev}
        sigma = eta * torch.sqrt((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev))
        dir_xt = torch.sqrt(torch.clamp(1 - a_prev - sigma ** 2, min=0.0)) * eps
        noise = sigma * torch.randn_like(x_t) if eta > 0 else 0.0

        return torch.sqrt(a_prev) * x0_pred + dir_xt + noise

    @torch.no_grad()
    def ddim_sample(self, shape, y=None, guidance_scale=1.0, ddim_steps=50, eta=0.0):
        """
        Replacement sampling loop: walk `ddim_steps` timesteps (a subsequence
        of 0..T-1) instead of all T, calling `ddim_step` at each one.
        """
        device = self.device
        times = torch.linspace(self.scheduler.T - 1, 0, steps=ddim_steps).round().long()
        times = times.to(device)

        x = torch.randn(shape, device=device)
        for i in range(len(times)):
            t = int(times[i].item())
            t_prev = int(times[i + 1].item()) if i < len(times) - 1 else -1
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)

            eps = self._predict_noise(x, t_batch, y, guidance_scale)
            x = self.ddim_step(x, eps, t, t_prev, eta)
        return x

    # ---------------------------------------------------------------- #
    # helper: classifier-free-guidance noise prediction (reused, unchanged)
    # ---------------------------------------------------------------- #
    def _predict_noise(self, x_t, t, y, guidance_scale):
        if y is not None and self.model.num_classes is not None and guidance_scale != 1.0:
            null_y = torch.full_like(y, self.model.num_classes)
            eps_cond = self.model(x_t, t, y)
            eps_uncond = self.model(x_t, t, null_y)
            return eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        return self.model(x_t, t, y)
