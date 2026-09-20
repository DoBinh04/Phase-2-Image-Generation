"""
Minimal U-Net for epsilon-prediction (predicting the noise added at step t).

Building blocks:
  - SinusoidalPosEmb : sinusoidal timestep embedding (Transformer-style)
  - ResBlock          : GroupNorm -> SiLU -> Conv, with the time (+ class)
                         embedding injected additively after the first conv
  - SelfAttention     : plain spatial self-attention, used once at the
                         bottleneck (helps global coherence, still "minimal")
  - Downsample/Upsample: strided conv / transposed conv

The U-Net optionally supports class conditioning for classifier-free
guidance (bonus part): pass `num_classes`, and an extra "null" class index
(== num_classes) is reserved as the unconditional token.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        freq = math.log(10000) / max(half_dim - 1, 1)
        freq = torch.exp(torch.arange(half_dim, device=device).float() * -freq)
        args = t.float()[:, None] * freq[None, :]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        if self.dim % 2 == 1:  # zero-pad if dim is odd
            emb = F.pad(emb, (0, 1))
        return emb


class ResBlock(nn.Module):
    """Two conv blocks with GroupNorm+SiLU, a residual connection, and the
    time/class embedding injected as a per-channel bias after the first conv."""

    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_proj = nn.Linear(time_emb_dim, out_ch)
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_ch), nn.SiLU(), nn.Conv2d(in_ch, out_ch, 3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_ch), nn.SiLU(), nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.block1(x)
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    """Standard non-local spatial self-attention (single head)."""

    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, kernel_size=1)
        self.proj = nn.Conv2d(ch, ch, kernel_size=1)
        self.scale = ch ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, C, H * W).permute(1, 0, 3, 2)  # (3, B, HW, C)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = torch.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)  # (B, HW, HW)
        out = attn @ v                                                       # (B, HW, C)
        out = out.permute(0, 2, 1).reshape(B, C, H, W)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.ConvTranspose2d(ch, ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class UNet(nn.Module):
    """
    3-level U-Net (32x32 -> 16x16 -> 8x8 bottleneck -> 16x16 -> 32x32).
    Works for MNIST (padded to 32x32, in_channels=1) or CIFAR-10
    (32x32 natively, in_channels=3).

    If `num_classes` is given, the model becomes class-conditional: it
    expects an integer label tensor `y` of shape (B,) at every forward call
    (values 0..num_classes-1 for a real class, or num_classes for the
    "unconditional" token used in classifier-free-guidance training/sampling).
    """

    def __init__(self, in_channels=1, base_ch=64, time_emb_dim=256, num_classes=None):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_ch),
            nn.Linear(base_ch, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        if num_classes is not None:
            # +1 slot reserved as the "null" / unconditional class for CFG
            self.label_emb = nn.Embedding(num_classes + 1, time_emb_dim)

        ch1, ch2, ch3 = base_ch, base_ch * 2, base_ch * 4

        self.init_conv = nn.Conv2d(in_channels, ch1, 3, padding=1)  # 32x32

        # ---- Encoder ----
        self.down1a = ResBlock(ch1, ch1, time_emb_dim)
        self.down1b = ResBlock(ch1, ch1, time_emb_dim)
        self.pool1 = Downsample(ch1)                                 # -> 16x16

        self.down2a = ResBlock(ch1, ch2, time_emb_dim)
        self.down2b = ResBlock(ch2, ch2, time_emb_dim)
        self.pool2 = Downsample(ch2)                                 # -> 8x8

        # ---- Bottleneck ----
        self.mid1 = ResBlock(ch2, ch3, time_emb_dim)
        self.mid_attn = SelfAttention(ch3)
        self.mid2 = ResBlock(ch3, ch3, time_emb_dim)

        # ---- Decoder ----
        self.up2 = Upsample(ch3)                                     # -> 16x16
        self.up2a = ResBlock(ch3 + ch2, ch2, time_emb_dim)
        self.up2b = ResBlock(ch2, ch2, time_emb_dim)

        self.up1 = Upsample(ch2)                                     # -> 32x32
        self.up1a = ResBlock(ch2 + ch1, ch1, time_emb_dim)
        self.up1b = ResBlock(ch1, ch1, time_emb_dim)

        self.final = nn.Sequential(
            nn.GroupNorm(8, ch1),
            nn.SiLU(),
            nn.Conv2d(ch1, in_channels, 3, padding=1),
        )

    def forward(self, x, t, y=None):
        """
        x: (B, C, H, W) noisy image batch
        t: (B,) integer timesteps
        y: (B,) integer class labels, or None for unconditional model
        """
        t_emb = self.time_mlp(t)

        if self.num_classes is not None:
            if y is None:
                # unconditional forward pass -> use the null token
                y = torch.full((x.shape[0],), self.num_classes, device=x.device, dtype=torch.long)
            t_emb = t_emb + self.label_emb(y)

        x0 = self.init_conv(x)                  # (B, ch1, 32, 32)

        x1 = self.down1a(x0, t_emb)
        x1 = self.down1b(x1, t_emb)              # skip1: (B, ch1, 32, 32)
        h = self.pool1(x1)                       # (B, ch1, 16, 16)

        x2 = self.down2a(h, t_emb)
        x2 = self.down2b(x2, t_emb)              # skip2: (B, ch2, 16, 16)
        h = self.pool2(x2)                       # (B, ch2, 8, 8)

        h = self.mid1(h, t_emb)                  # (B, ch3, 8, 8)
        h = self.mid_attn(h)
        h = self.mid2(h, t_emb)

        h = self.up2(h)                          # (B, ch3, 16, 16)
        h = torch.cat([h, x2], dim=1)             # (B, ch3+ch2, 16, 16)
        h = self.up2a(h, t_emb)
        h = self.up2b(h, t_emb)                   # (B, ch2, 16, 16)

        h = self.up1(h)                          # (B, ch2, 32, 32)
        h = torch.cat([h, x1], dim=1)             # (B, ch2+ch1, 32, 32)
        h = self.up1a(h, t_emb)
        h = self.up1b(h, t_emb)                   # (B, ch1, 32, 32)

        return self.final(h)                     # (B, in_channels, 32, 32)
