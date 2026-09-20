"""
verify_vae.py — sanity checks for vae.py
=========================================

Chạy độc lập, KHÔNG cần tải dataset (MNIST/CelebA). Dùng dữ liệu giả
(random tensor) để kiểm tra model đúng về mặt kỹ thuật trước khi bạn
tốn thời gian train thật.

Chạy:
    python verify_vae.py

Các phép kiểm tra:
  1. Shape test      — encoder/decoder cho ra đúng kích thước, cho cả
                        cấu hình MNIST (1x32x32) và CelebA (3x64x64).
  2. Backward test    — loss.backward() chạy được, không NaN/Inf,
                         và mọi tham số đều nhận được gradient.
  3. KL sanity        — nếu mu=0, logvar=0 (q(z|x) = N(0,I) = prior)
                         thì KL phải xấp xỉ 0.
  4. Reparam sanity   — với cùng mu/logvar, hai lần sample phải cho
                         ra z khác nhau (có tính ngẫu nhiên) nhưng
                         cùng mean nếu lấy trung bình nhiều mẫu.
  5. Overfit test     — cho model học thuộc lòng 1 batch nhỏ trong
                         vài trăm bước; loss phải giảm mạnh và
                         reconstruction phải gần giống ảnh gốc.
                         Đây là bằng chứng mạnh nhất rằng gradient
                         thực sự tối ưu đúng hướng.
"""

import torch
import torch.nn.functional as F

from vae import VAE, vae_loss


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    if not cond:
        raise AssertionError(name)


def shape_and_backward_test(in_channels, img_size, tag):
    print(f"\n--- {tag}: in_channels={in_channels}, img_size={img_size} ---")
    torch.manual_seed(0)
    model = VAE(in_channels=in_channels, img_size=img_size, base_ch=16, latent_dim=32)
    x = torch.rand(4, in_channels, img_size, img_size)

    x_recon, mu, logvar = model(x)

    check(f"{tag}: recon shape == input shape", x_recon.shape == x.shape)
    check(f"{tag}: mu shape == (4, latent_dim)", mu.shape == (4, 32))
    check(f"{tag}: logvar shape == (4, latent_dim)", logvar.shape == (4, 32))
    check(f"{tag}: recon output in [0,1]", (x_recon.min() >= 0) and (x_recon.max() <= 1))

    loss, recon, kl = vae_loss(x_recon, x, mu, logvar, beta=1.0, recon_type="bce")
    check(f"{tag}: loss is finite", torch.isfinite(loss).item())

    loss.backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    n_params = sum(1 for _ in model.parameters())
    check(f"{tag}: every parameter received a gradient", len(grad_norms) == n_params)
    check(f"{tag}: gradients are finite and non-zero overall",
          all(g == g and g != float("inf") for g in grad_norms) and sum(grad_norms) > 0)

    samples = model.sample(5, torch.device("cpu"))
    check(f"{tag}: sample() shape correct", samples.shape == (5, in_channels, img_size, img_size))


def kl_zero_test():
    print("\n--- KL sanity: q(z|x) == prior N(0,I) => KL ~= 0 ---")
    mu = torch.zeros(8, 16)
    logvar = torch.zeros(8, 16)  # log(sigma^2) = 0 -> sigma = 1
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)
    check("KL(N(0,I) || N(0,I)) ~= 0", abs(kl.item()) < 1e-5)


def reparam_randomness_test():
    print("\n--- Reparameterization: sampling is stochastic but unbiased ---")
    torch.manual_seed(1)
    mu = torch.zeros(1, 4)
    logvar = torch.zeros(1, 4)

    z1 = VAE.reparameterize(mu, logvar)
    z2 = VAE.reparameterize(mu, logvar)
    check("two samples from the same mu/logvar differ", not torch.allclose(z1, z2))

    samples = torch.stack([VAE.reparameterize(mu, logvar) for _ in range(2000)])
    empirical_mean = samples.mean().item()
    check(f"empirical mean over 2000 samples ~= 0 (got {empirical_mean:.3f})",
          abs(empirical_mean) < 0.1)


def overfit_test(steps=300, tag="MNIST-like"):
    print(f"\n--- Overfit test ({tag}): can the model memorize one small batch? ---")
    torch.manual_seed(0)
    model = VAE(in_channels=1, img_size=32, base_ch=16, latent_dim=32)
    x = torch.rand(8, 1, 32, 32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    for step in range(steps):
        opt.zero_grad()
        x_recon, mu, logvar = model(x)
        # beta rất nhỏ để test tập trung vào khả năng tái tạo (recon),
        # đúng với mục đích overfit-test (không phải test khái quát hoá).
        loss, recon, kl = vae_loss(x_recon, x, mu, logvar, beta=0.01, recon_type="mse")
        loss.backward()
        opt.step()
        losses.append(loss.item())

    first, last = losses[0], losses[-1]
    print(f"    loss[0]={first:.4f}  ->  loss[{steps-1}]={last:.4f}")
    check("overfit loss decreases substantially (>90% drop)", last < first * 0.1)

    with torch.no_grad():
        x_recon, _, _ = model(x)
        mse = F.mse_loss(x_recon, x).item()
    print(f"    final pixel-wise MSE on memorized batch: {mse:.5f}")
    check("final reconstruction MSE is small (<0.01)", mse < 0.01)


if __name__ == "__main__":
    shape_and_backward_test(in_channels=1, img_size=32, tag="MNIST-config")
    shape_and_backward_test(in_channels=3, img_size=64, tag="CelebA-config")
    kl_zero_test()
    reparam_randomness_test()
    overfit_test()

    print("\nALL CHECKS PASSED ✅ — model kiến trúc, gradient, và khả năng học đều ổn.")
