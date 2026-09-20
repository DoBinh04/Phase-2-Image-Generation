"""
Generate samples from a trained checkpoint.

Unconditional:
    python sample.py --ckpt checkpoints/ddpm_last.pt --n 64 --out samples/grid.png

Class-conditional (bonus), one row per digit 0-9, with classifier-free guidance:
    python sample.py --ckpt checkpoints/ddpm_last.pt --conditional \
        --n_per_class 8 --guidance_scale 3.0 --out samples/class_conditional_grid.png
"""
import argparse
import torch
from torchvision.utils import save_image

from scheduler import NoiseScheduler
from model import UNet
from diffusion import GaussianDiffusion


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--conditional", action="store_true")
    p.add_argument("--n", type=int, default=64, help="num samples (unconditional mode)")
    p.add_argument("--n_per_class", type=int, default=8, help="samples per digit (conditional mode)")
    p.add_argument("--guidance_scale", type=float, default=3.0)
    p.add_argument("--sampler", type=str, default="ddim", choices=["ddim", "ddpm"])
    p.add_argument("--ddim_steps", type=int, default=50)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--schedule", type=str, default="cosine", choices=["linear", "cosine"])
    p.add_argument("--base_ch", type=int, default=64)
    p.add_argument("--out", type=str, default="samples/grid.png")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    scheduler = NoiseScheduler(timesteps=args.timesteps, schedule=args.schedule, device=device)
    num_classes = 10 if args.conditional else None
    model = UNet(in_channels=1, base_ch=args.base_ch, num_classes=num_classes).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    diffusion = GaussianDiffusion(model, scheduler, device)

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

    with torch.no_grad():
        if args.sampler == "ddim":
            imgs = diffusion.ddim_sample(shape, y=y, guidance_scale=guidance, ddim_steps=args.ddim_steps)
        else:
            imgs = diffusion.sample(shape, y=y, guidance_scale=guidance)

    imgs = (imgs.clamp(-1, 1) + 1) / 2  # [-1,1] -> [0,1]
    save_image(imgs, args.out, nrow=nrow)
    print(f"Saved {shape[0]} samples to {args.out}")
    if args.conditional:
        print(f"Each row = 1 digit class (0-9 top to bottom), {args.n_per_class} samples per row, "
              f"guidance_scale={args.guidance_scale}")


if __name__ == "__main__":
    main()
