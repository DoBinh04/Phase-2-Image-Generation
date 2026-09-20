"""
Train a (class-conditional) DDPM on MNIST.

Usage:
    python train.py --epochs 20 --batch_size 128 --schedule cosine --conditional
    python train.py --epochs 20 --schedule linear            # unconditional

Checkpoints go to ./checkpoints/, sample grids go to ./samples/.
"""
import argparse
import os

import torch
from torchvision.utils import save_image

from scheduler import NoiseScheduler
from model import UNet
from diffusion import GaussianDiffusion
from data import get_mnist_dataloader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--schedule", type=str, default="cosine", choices=["linear", "cosine"])
    p.add_argument("--base_ch", type=int, default=64)
    p.add_argument("--conditional", action="store_true", help="train class-conditional model (bonus)")
    p.add_argument("--p_uncond", type=float, default=0.1, help="label dropout prob for CFG training")
    p.add_argument("--sample_every", type=int, default=1, help="epochs between sample grids")
    p.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    p.add_argument("--sample_dir", type=str, default="./samples")
    p.add_argument("--data_root", type=str, default="./data")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)

    # ---- data ----
    loader = get_mnist_dataloader(batch_size=args.batch_size, root=args.data_root)

    # ---- scheduler + model + diffusion wrapper ----
    scheduler = NoiseScheduler(timesteps=args.timesteps, schedule=args.schedule, device=device)
    num_classes = 10 if args.conditional else None
    model = UNet(in_channels=1, base_ch=args.base_ch, num_classes=num_classes).to(device)
    diffusion = GaussianDiffusion(model, scheduler, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Conditional: {args.conditional} | schedule: {args.schedule} | T: {args.timesteps}")

    step = 0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for x0, y in loader:
            x0 = x0.to(device)
            y = y.to(device) if args.conditional else None

            t = torch.randint(0, scheduler.T, (x0.shape[0],), device=device).long()

            loss = diffusion.p_losses(x0, t, y=y, p_uncond=args.p_uncond)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            step += 1
            if step % 100 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")

        avg_loss = running_loss / len(loader)
        print(f"== epoch {epoch} done | avg loss {avg_loss:.4f} ==")

        torch.save(
            {"model": model.state_dict(), "epoch": epoch, "args": vars(args)},
            os.path.join(args.ckpt_dir, "ddpm_last.pt"),
        )

        if (epoch + 1) % args.sample_every == 0:
            model.eval()
            _quick_sample_grid(diffusion, args, device, epoch)


@torch.no_grad()
def _quick_sample_grid(diffusion, args, device, epoch, n_per_class=8):
    """Draw a quick sample grid during training so you can watch progress."""
    if args.conditional:
        y = torch.arange(10, device=device).repeat_interleave(n_per_class)
        shape = (y.shape[0], 1, 32, 32)
        imgs = diffusion.ddim_sample(shape, y=y, guidance_scale=3.0, ddim_steps=50)
        nrow = n_per_class
    else:
        shape = (64, 1, 32, 32)
        imgs = diffusion.ddim_sample(shape, y=None, guidance_scale=1.0, ddim_steps=50)
        nrow = 8

    imgs = (imgs.clamp(-1, 1) + 1) / 2  # back to [0, 1]
    save_image(imgs, os.path.join(args.sample_dir, f"epoch_{epoch:03d}.png"), nrow=nrow)


if __name__ == "__main__":
    main()
