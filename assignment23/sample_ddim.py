"""
Assignment 2.3 — Reuse the model trained in Assignment 2, replace the
sampling loop with DDIM.

This script does NOT train anything. It loads the checkpoint produced by
Assignment 2's training run (same UNet architecture, same weights) and
generates samples using `GaussianDiffusion.ddim_sample` instead of the
original T=1000-step `GaussianDiffusion.sample` (DDPM) loop.

Usage:
    # unconditional
    python sample_ddim.py --ckpt path/to/ddpm_last.pt --n 64 --ddim_steps 50

    # class-conditional (if Assignment 2's model was trained with --conditional)
    python sample_ddim.py --ckpt path/to/ddpm_last.pt --conditional \
        --n_per_class 8 --guidance_scale 3.0 --ddim_steps 50 --eta 0.0

    # also save the old DDPM loop's output for a quick visual comparison
    python sample_ddim.py --ckpt path/to/ddpm_last.pt --compare_ddpm
"""
import argparse
import os

import torch
from torchvision.utils import save_image

from scheduler import NoiseScheduler
from model import UNet
from diffusion import GaussianDiffusion


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, help="checkpoint trained in Assignment 2")
    p.add_argument("--conditional", action="store_true", help="set if the checkpoint was trained class-conditionally")
    p.add_argument("--n", type=int, default=64, help="num samples (unconditional mode)")
    p.add_argument("--n_per_class", type=int, default=8, help="samples per digit (conditional mode)")
    p.add_argument("--guidance_scale", type=float, default=3.0)
    p.add_argument("--timesteps", type=int, default=1000, help="T used when the model was trained")
    p.add_argument("--schedule", type=str, default="cosine", choices=["linear", "cosine"])
    p.add_argument("--base_ch", type=int, default=64)
    p.add_argument("--ddim_steps", type=int, default=50)
    p.add_argument("--eta", type=float, default=0.0, help="0 = deterministic DDIM, 1 = DDPM-like stochastic")
    p.add_argument("--compare_ddpm", action="store_true", help="also run the original T-step DDPM loop for comparison")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=str, default="samples")
    return p.parse_args()


def build_model(args, device):
    scheduler = NoiseScheduler(timesteps=args.timesteps, schedule=args.schedule, device=device)
    num_classes = 10 if args.conditional else None
    model = UNet(in_channels=1, base_ch=args.base_ch, num_classes=num_classes).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    return GaussianDiffusion(model, scheduler, device), num_classes


def to_image(x):
    return (x.clamp(-1, 1) + 1) / 2


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    os.makedirs(args.out_dir, exist_ok=True)

    diffusion, num_classes = build_model(args, device)

    if args.conditional:
        y = torch.arange(10, device=device).repeat_interleave(args.n_per_class)
        shape = (y.shape[0], 1, 32, 32)
        nrow = args.n_per_class
        guidance = args.guidance_scale
    else:
        y = None
        shape = (args.n, 1, 32, 32)
        nrow = int(shape[0] ** 0.5)
        guidance = 1.0

    torch.manual_seed(args.seed)
    x_T = torch.randn(shape, device=device)

    # ---- REPLACEMENT sampling loop (this assignment) ----
    with torch.no_grad():
        x_start = x_T.clone()
        # ddim_sample draws its own x_T internally; to make the optional
        # DDPM comparison below fully apples-to-apples we instead re-run its
        # per-step logic here on the same x_start.
        times = torch.linspace(diffusion.scheduler.T - 1, 0, steps=args.ddim_steps).round().long().to(device)
        x = x_start
        for i in range(len(times)):
            t = int(times[i].item())
            t_prev = int(times[i + 1].item()) if i < len(times) - 1 else -1
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            eps = diffusion._predict_noise(x, t_batch, y, guidance)
            x = diffusion.ddim_step(x, eps, t, t_prev, args.eta)
        ddim_imgs = x

    ddim_path = os.path.join(args.out_dir, f"ddim_{args.ddim_steps}steps_eta{args.eta}.png")
    save_image(to_image(ddim_imgs), ddim_path, nrow=nrow)
    print(f"Saved DDIM samples ({args.ddim_steps} steps, eta={args.eta}) -> {ddim_path}")

    # ---- optional: original DDPM loop, same x_T, same weights ----
    if args.compare_ddpm:
        with torch.no_grad():
            x = x_T.clone()
            for i in reversed(range(diffusion.scheduler.T)):
                t = torch.full((shape[0],), i, device=device, dtype=torch.long)
                x = diffusion.p_sample(x, t, i, y=y, guidance_scale=guidance)
            ddpm_imgs = x

        ddpm_path = os.path.join(args.out_dir, "ddpm_1000steps.png")
        save_image(to_image(ddpm_imgs), ddpm_path, nrow=nrow)
        print(f"Saved DDPM baseline samples (1000 steps) -> {ddpm_path}")

        side_by_side = torch.cat([to_image(ddpm_imgs), to_image(ddim_imgs)], dim=0)
        side_path = os.path.join(args.out_dir, "ddpm_vs_ddim.png")
        save_image(side_by_side, side_path, nrow=nrow)
        print(f"Saved side-by-side (row1=DDPM/1000, row2=DDIM/{args.ddim_steps}) -> {side_path}")


if __name__ == "__main__":
    main()
