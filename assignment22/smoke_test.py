"""
Fast sanity check that doesn't need any dataset download.
Verifies: scheduler math, q_sample shape, one training step (loss.backward),
and one short DDIM sampling pass, for both unconditional and conditional models.

Run:
    python smoke_test.py
"""
import torch
from scheduler import NoiseScheduler
from model import UNet
from diffusion import GaussianDiffusion


def run(conditional: bool, schedule: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B = 4
    scheduler = NoiseScheduler(timesteps=200, schedule=schedule, device=device)
    num_classes = 10 if conditional else None
    model = UNet(in_channels=1, base_ch=32, num_classes=num_classes).to(device)
    diffusion = GaussianDiffusion(model, scheduler, device)

    x0 = torch.randn(B, 1, 32, 32, device=device)
    y = torch.randint(0, 10, (B,), device=device) if conditional else None
    t = torch.randint(0, scheduler.T, (B,), device=device)

    # forward process
    x_t, noise = diffusion.q_sample(x0, t)
    assert x_t.shape == x0.shape

    # training step
    loss = diffusion.p_losses(x0, t, y=y)
    loss.backward()
    assert torch.isfinite(loss)
    print(f"[conditional={conditional}, schedule={schedule}] loss={loss.item():.4f}  OK")

    # sampling (DDPM, short)
    y_sample = torch.randint(0, 10, (B,), device=device) if conditional else None
    out = diffusion.sample((B, 1, 32, 32), y=y_sample, guidance_scale=3.0 if conditional else 1.0)
    assert out.shape == (B, 1, 32, 32)
    assert torch.isfinite(out).all()
    print(f"[conditional={conditional}] DDPM sample shape {tuple(out.shape)}  OK")

    # sampling (DDIM, short)
    out2 = diffusion.ddim_sample((B, 1, 32, 32), y=y_sample, guidance_scale=3.0 if conditional else 1.0, ddim_steps=10)
    assert out2.shape == (B, 1, 32, 32)
    assert torch.isfinite(out2).all()
    print(f"[conditional={conditional}] DDIM sample shape {tuple(out2.shape)}  OK")


if __name__ == "__main__":
    for schedule in ["linear", "cosine"]:
        run(conditional=False, schedule=schedule)
        run(conditional=True, schedule=schedule)
    print("\nAll smoke tests passed.")
