"""
Gaussian diffusion process: ties the NoiseScheduler + UNet together.

  - q_sample(x0, t, noise)   : forward process, closed-form x_t from x_0
  - p_losses(x0, t, y)       : training loss (predict-the-noise MSE),
                                with random label dropout for classifier-free
                                guidance when the model is class-conditional
  - p_sample / sample        : DDPM ancestral sampling (T steps)
  - ddim_sample              : faster deterministic/semi-deterministic
                                DDIM sampling (fewer steps), with optional
                                classifier-free guidance
"""
import torch
import torch.nn.functional as F


class GaussianDiffusion:
    def __init__(self, model, scheduler, device):
        self.model = model
        self.scheduler = scheduler
        self.device = device

    # ---------------------------------------------------------------- #
    # 2. Forward process
    # ---------------------------------------------------------------- #
    def q_sample(self, x0, t, noise=None):
        """
        Sample x_t ~ q(x_t | x_0) in closed form:
            x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ac = self.scheduler.extract(self.scheduler.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_omac = self.scheduler.extract(self.scheduler.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        x_t = sqrt_ac * x0 + sqrt_omac * noise
        return x_t, noise

    # ---------------------------------------------------------------- #
    # 4. Training loss
    # ---------------------------------------------------------------- #
    def p_losses(self, x0, t, y=None, p_uncond=0.1):
        """
        1. sample noise
        2. add noise to x0 at step t  -> x_t
        3. predict the noise with the U-Net
        4. MSE(predicted_noise, true_noise)

        If the model is class-conditional (`y` is not None), randomly
        replace a fraction `p_uncond` of the labels with the null token so
        the same network learns both the conditional and unconditional
        score (classifier-free guidance training).
        """
        noise = torch.randn_like(x0)
        x_t, noise = self.q_sample(x0, t, noise)

        if y is not None and self.model.num_classes is not None and p_uncond > 0:
            drop_mask = torch.rand(y.shape[0], device=y.device) < p_uncond
            y = y.clone()
            y[drop_mask] = self.model.num_classes  # null / unconditional token

        pred_noise = self.model(x_t, t, y)
        return F.mse_loss(pred_noise, noise)

    # ---------------------------------------------------------------- #
    # 5. Sampling — DDPM (full T-step ancestral sampling)
    # ---------------------------------------------------------------- #
    @torch.no_grad()
    def p_sample(self, x_t, t, t_index, y=None, guidance_scale=1.0):
        """One reverse step: x_t -> x_{t-1}."""
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
        """Full T-step DDPM sampling loop, starting from pure Gaussian noise."""
        device = self.device
        x = torch.randn(shape, device=device)
        for i in reversed(range(self.scheduler.T)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t, i, y=y, guidance_scale=guidance_scale)
        return x

    # ---------------------------------------------------------------- #
    # 5b. Sampling — DDIM (fewer steps, deterministic when eta=0)
    # ---------------------------------------------------------------- #
    @torch.no_grad()
    def ddim_sample(self, shape, y=None, guidance_scale=1.0, ddim_steps=50, eta=0.0):
        device = self.device
        alphas_cumprod = self.scheduler.alphas_cumprod
        # build as float then round -> avoids dtype=torch.long linspace, which
        # some older PyTorch builds don't support
        times = torch.linspace(self.scheduler.T - 1, 0, steps=ddim_steps)
        times = times.round().long().to(device)

        x = torch.randn(shape, device=device)

        for i in range(len(times)):
            t = times[i]
            t_batch = torch.full((shape[0],), t.item(), device=device, dtype=torch.long)

            eps = self._predict_noise(x, t_batch, y, guidance_scale)

            a_t = alphas_cumprod[t]
            a_prev = alphas_cumprod[times[i + 1]] if i < len(times) - 1 else torch.tensor(1.0, device=device)

            x0_pred = (x - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t)
            x0_pred = x0_pred.clamp(-1.0, 1.0)

            sigma = eta * torch.sqrt((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev))
            dir_xt = torch.sqrt(torch.clamp(1 - a_prev - sigma ** 2, min=0.0)) * eps
            noise = sigma * torch.randn_like(x) if eta > 0 else 0.0

            x = torch.sqrt(a_prev) * x0_pred + dir_xt + noise

        return x

    # ---------------------------------------------------------------- #
    # helper: classifier-free-guidance noise prediction
    # ---------------------------------------------------------------- #
    def _predict_noise(self, x_t, t, y, guidance_scale):
        if y is not None and self.model.num_classes is not None and guidance_scale != 1.0:
            null_y = torch.full_like(y, self.model.num_classes)
            eps_cond = self.model(x_t, t, y)
            eps_uncond = self.model(x_t, t, null_y)
            # classifier-free guidance extrapolation
            return eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        return self.model(x_t, t, y)
