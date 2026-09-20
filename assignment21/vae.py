"""
Assignment 2.1 — Variational Autoencoder (VAE) in pure PyTorch
================================================================

    Encoder: Conv layers            -> mu, log(sigma^2)
    Reparameterization: z = mu + eps * sigma
    Decoder: ConvTranspose layers   -> reconstructed image
    Loss = Reconstruction Loss (BCE/MSE) + beta * KL Divergence

Supports two datasets, switched with --dataset:
    * mnist   -> 1x32x32  (grayscale, MNIST is zero-padded 28->32)
    * celeba  -> 3x64x64  (RGB, center-cropped + resized)

Usage
-----
    python vae.py --dataset mnist  --epochs 15 --beta 1.0
    python vae.py --dataset celeba --epochs 30 --beta 1.0 \
                  --data-root /path/to/celeba

CelebA must already be downloaded (torchvision's CelebA downloader is
frequently rate-limited by Google Drive), laid out as:
    <data-root>/celeba/img_align_celeba/*.jpg
    <data-root>/celeba/list_eval_partition.txt
    <data-root>/celeba/list_attr_celeba.txt
i.e. exactly the structure torchvision.datasets.CelebA expects.

Everything (model, reparameterization trick, loss, training loop,
sampling / reconstruction grids) is implemented from scratch with
torch.nn — no pretrained components, no external VAE libraries.
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class Encoder(nn.Module):
    """Conv stack that downsamples the image to a spatial bottleneck,
    then two linear heads produce mu and log(sigma^2)."""

    def __init__(self, in_channels: int, img_size: int, base_ch: int, latent_dim: int):
        super().__init__()
        # Each block halves H and W: img_size -> img_size/2 -> .../4 -> .../8 -> .../16
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, 4, stride=2, padding=1),      # H/2
            nn.BatchNorm2d(base_ch),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_ch, base_ch * 2, 4, stride=2, padding=1),      # H/4
            nn.BatchNorm2d(base_ch * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_ch * 2, base_ch * 4, 4, stride=2, padding=1),  # H/8
            nn.BatchNorm2d(base_ch * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_ch * 4, base_ch * 8, 4, stride=2, padding=1),  # H/16
            nn.BatchNorm2d(base_ch * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.feat_size = img_size // 16          # spatial size after 4 stride-2 convs
        self.flat_dim = base_ch * 8 * self.feat_size * self.feat_size

        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)

    def forward(self, x):
        h = self.conv(x)
        h = h.flatten(1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    """Linear projection back to a spatial map, then ConvTranspose layers
    that upsample to the original image resolution."""

    def __init__(self, out_channels: int, img_size: int, base_ch: int, latent_dim: int):
        super().__init__()
        self.feat_size = img_size // 16
        self.base_ch = base_ch

        self.fc = nn.Linear(latent_dim, base_ch * 8 * self.feat_size * self.feat_size)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 4, stride=2, padding=1),  # H/8
            nn.BatchNorm2d(base_ch * 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, stride=2, padding=1),  # H/4
            nn.BatchNorm2d(base_ch * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(base_ch * 2, base_ch, 4, stride=2, padding=1),      # H/2
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(base_ch, out_channels, 4, stride=2, padding=1),     # H
            # Sigmoid squashes to [0, 1] so BCE loss is valid pixel-wise.
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, self.base_ch * 8, self.feat_size, self.feat_size)
        return self.deconv(h)


class VAE(nn.Module):
    def __init__(self, in_channels=1, img_size=32, base_ch=32, latent_dim=128):
        super().__init__()
        self.encoder = Encoder(in_channels, img_size, base_ch, latent_dim)
        self.decoder = Decoder(in_channels, img_size, base_ch, latent_dim)
        self.latent_dim = latent_dim

    @staticmethod
    def reparameterize(mu, logvar):
        """z = mu + eps * sigma, eps ~ N(0, I). Sampling is moved onto eps
        so gradients can flow through mu / logvar (the 'reparameterization
        trick')."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

    @torch.no_grad()
    def sample(self, n, device):
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decoder(z)


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #
def vae_loss(x_recon, x, mu, logvar, beta=1.0, recon_type="bce"):
    """Loss = Reconstruction + beta * KL(q(z|x) || N(0, I)).

    Both terms are summed over pixels/dims and averaged over the batch,
    which is the standard convention (keeps the two terms on a comparable
    scale regardless of batch size).
    """
    B = x.size(0)

    if recon_type == "bce":
        recon = F.binary_cross_entropy(x_recon, x, reduction="sum") / B
    elif recon_type == "mse":
        recon = F.mse_loss(x_recon, x, reduction="sum") / B
    else:
        raise ValueError("recon_type must be 'bce' or 'mse'")

    # Closed-form KL divergence between N(mu, sigma^2) and N(0, I):
    #   KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / B

    return recon + beta * kl, recon, kl


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def get_dataloaders(dataset: str, data_root: str, batch_size: int, img_size: int):
    if dataset == "mnist":
        tfm = transforms.Compose([
            transforms.Resize(img_size),   # 28 -> 32
            transforms.ToTensor(),         # [0, 1], shape (1, H, W)
        ])
        train_ds = datasets.MNIST(data_root, train=True, download=True, transform=tfm)
        test_ds = datasets.MNIST(data_root, train=False, download=True, transform=tfm)
        in_channels = 1

    elif dataset == "celeba":
        tfm = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize(img_size),   # -> 64x64
            transforms.ToTensor(),         # [0, 1], shape (3, H, W)
        ])
        # download=False by default: torchvision's CelebA auto-download is
        # frequently rate-limited; pass --download to try it anyway.
        train_ds = datasets.CelebA(data_root, split="train", download=False, transform=tfm)
        test_ds = datasets.CelebA(data_root, split="test", download=False, transform=tfm)
        in_channels = 3

    else:
        raise ValueError("dataset must be 'mnist' or 'celeba'")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=2, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    return train_loader, test_loader, in_channels


# --------------------------------------------------------------------------- #
# Train / eval loops
# --------------------------------------------------------------------------- #
def train_one_epoch(model, loader, optimizer, device, beta, recon_type, epoch, log_every=100):
    model.train()
    totals = {"loss": 0.0, "recon": 0.0, "kl": 0.0}

    for i, (x, _) in enumerate(loader):
        x = x.to(device, non_blocking=True)

        optimizer.zero_grad()
        x_recon, mu, logvar = model(x)
        loss, recon, kl = vae_loss(x_recon, x, mu, logvar, beta=beta, recon_type=recon_type)
        loss.backward()
        optimizer.step()

        totals["loss"] += loss.item()
        totals["recon"] += recon.item()
        totals["kl"] += kl.item()

        if i % log_every == 0:
            print(f"epoch {epoch} [{i:4d}/{len(loader)}] "
                  f"loss={loss.item():.2f}  recon={recon.item():.2f}  kl={kl.item():.2f}")

    n = len(loader)
    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def evaluate(model, loader, device, beta, recon_type):
    model.eval()
    totals = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        x_recon, mu, logvar = model(x)
        loss, recon, kl = vae_loss(x_recon, x, mu, logvar, beta=beta, recon_type=recon_type)
        totals["loss"] += loss.item()
        totals["recon"] += recon.item()
        totals["kl"] += kl.item()
    n = len(loader)
    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def save_samples(model, test_loader, device, out_dir, epoch, n=64):
    os.makedirs(out_dir, exist_ok=True)

    # (a) Prior samples: z ~ N(0, I) -> decoder
    samples = model.sample(n, device)
    save_image(samples, os.path.join(out_dir, f"samples_epoch{epoch:03d}.png"), nrow=8)

    # (b) Reconstructions: real x -> encoder -> z -> decoder, shown next to originals
    x, _ = next(iter(test_loader))
    x = x[:n // 2].to(device)
    x_recon, _, _ = model(x)
    comparison = torch.cat([x, x_recon], dim=0)
    save_image(comparison, os.path.join(out_dir, f"recon_epoch{epoch:03d}.png"), nrow=n // 2)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="Pure PyTorch VAE (MNIST / CelebA-64x64)")
    p.add_argument("--dataset", choices=["mnist", "celeba"], default="mnist")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--out-dir", default="./vae_out")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--base-ch", type=int, default=32)
    p.add_argument("--beta", type=float, default=1.0, help="weight on the KL term (beta-VAE)")
    p.add_argument("--recon", choices=["bce", "mse"], default="bce")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    img_size = 32 if args.dataset == "mnist" else 64
    train_loader, test_loader, in_channels = get_dataloaders(
        args.dataset, args.data_root, args.batch_size, img_size
    )

    model = VAE(in_channels=in_channels, img_size=img_size,
                base_ch=args.base_ch, latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(model)
    n_params = sum(p_.numel() for p_ in model.parameters())
    print(f"total parameters: {n_params:,}")

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(model, train_loader, optimizer, device,
                                       args.beta, args.recon, epoch)
        test_stats = evaluate(model, test_loader, device, args.beta, args.recon)
        print(f"== epoch {epoch} == "
              f"train loss={train_stats['loss']:.2f} "
              f"(recon={train_stats['recon']:.2f}, kl={train_stats['kl']:.2f})  |  "
              f"test loss={test_stats['loss']:.2f} "
              f"(recon={test_stats['recon']:.2f}, kl={test_stats['kl']:.2f})")

        save_samples(model, test_loader, device, args.out_dir, epoch)

    ckpt_path = os.path.join(args.out_dir, "vae_final.pt")
    torch.save({"model_state": model.state_dict(), "args": vars(args)}, ckpt_path)
    print(f"saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
