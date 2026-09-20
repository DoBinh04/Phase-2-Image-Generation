"""
Verification suite for the DDIM sampling loop (Assignment 2.3).

Answers the question "did I implement DDIM correctly?" with checks that
don't just eyeball generated digits -- they test the *update formula*
itself against known closed-form identities. Tests 1-2 need no trained
model at all (they isolate the math from model quality); Tests 3-4 are
optional empirical sanity checks that use a real checkpoint if you pass one.

Run:
    python verify_ddim.py                          # Test 1 + Test 2 only (no checkpoint needed)
    python verify_ddim.py --ckpt path/to/ddpm_last.pt [--conditional]   # + Test 3 + Test 4
"""
import argparse
import torch

from scheduler import NoiseScheduler
from model import UNet
from diffusion import GaussianDiffusion


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=None, help="optional: enables Test 3 and Test 4")
    p.add_argument("--conditional", action="store_true")
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--schedule", type=str, default="cosine", choices=["linear", "cosine"])
    p.add_argument("--base_ch", type=int, default=64)
    # Test 2 compares two algebraically equal expressions formed from
    # float32 cumulative products.  Near t=0 its posterior variance is very
    # small, so one float32 division/subtraction can produce ~1.7e-4 relative
    # error even though the absolute error is only a few e-9.
    p.add_argument("--tol", type=float, default=2e-4,
                   help="float32 relative tolerance for the algebraic checks")
    return p.parse_args()


# ===================================================================== #
# Test 1 (no model): DDIM with the EXACT forward-process noise
#         reconstructs the EXACT forward-process trajectory.
# ===================================================================== #
#
# Why this is a valid correctness test:
#   The forward process defines, in closed form,
#       x_t = sqrt(a_t) * x0 + sqrt(1 - a_t) * eps            (*)
#   for ANY t, using the SAME eps and x0.
#
#   Algebraically, if you feed that same exact eps into the DDIM update
#   (eta=0) starting from x_t, the predicted x0 is:
#       x0_pred = (x_t - sqrt(1-a_t) * eps) / sqrt(a_t)  =  x0   (exact, by (*))
#   and the DDIM step then computes:
#       x_s = sqrt(a_s) * x0_pred + sqrt(1 - a_s) * eps  =  sqrt(a_s) * x0 + sqrt(1-a_s) * eps
#   which is EXACTLY the closed-form definition (*) of x_s.
#
#   So: ddim_step(x_t, exact_eps, t, s, eta=0) must equal the closed-form
#   x_s to floating-point precision, for ANY pair t > s and ANY x0. If it
#   doesn't, the update formula is implemented wrong -- this has nothing to
#   do with the neural network.
def test_1_algebraic_trajectory_consistency(diffusion, tol):
    print("\n[Test 1] DDIM step reproduces the exact forward-process trajectory (eta=0, no model)")
    device = diffusion.device
    torch.manual_seed(0)
    B, C, H, W = 4, 1, 32, 32
    max_err = 0.0

    for trial in range(20):
        x0 = torch.rand(B, C, H, W, device=device) * 2 - 1     # keep inside [-1,1], like real data
        eps = torch.randn(B, C, H, W, device=device)

        t = torch.randint(200, diffusion.scheduler.T, (1,)).item()
        s = torch.randint(0, t, (1,)).item()                    # s < t, arbitrary gap (tests non-adjacent steps too)

        a_t = diffusion.scheduler.alphas_cumprod[t]
        a_s = diffusion.scheduler.alphas_cumprod[s]
        x_t = torch.sqrt(a_t) * x0 + torch.sqrt(1 - a_t) * eps
        x_s_expected = torch.sqrt(a_s) * x0 + torch.sqrt(1 - a_s) * eps

        x_s_actual = diffusion.ddim_step(x_t, eps, t, s, eta=0.0)

        err = (x_s_actual - x_s_expected).abs().max().item()
        max_err = max(max_err, err)

    ok = max_err < tol
    print(f"  max abs error over 20 random (t, s, x0, eps) trials: {max_err:.2e}  (tol={tol:.0e})")
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


# ===================================================================== #
# Test 2 (no model): DDIM's eta=1 noise variance matches the DDPM
#         posterior variance for ADJACENT timesteps.
# ===================================================================== #
#
# Why this is a valid correctness test:
#   DDIM defines sigma_t^2 = eta^2 * (1-a_prev)/(1-a_t) * (1 - a_t/a_prev).
#   For adjacent steps (t_prev = t-1), a_t / a_prev = alpha_t (the single-step
#   alpha), so (1 - a_t/a_prev) = beta_t, and with eta=1:
#       sigma_t^2 = (1 - a_{t-1})/(1 - a_t) * beta_t
#   which is EXACTLY the DDPM posterior variance formula
#   (scheduler.posterior_variance[t], used by the original p_sample loop).
#   If eta=1 doesn't reduce to this, the sigma formula is wrong.
def test_2_eta1_matches_ddpm_posterior_variance(diffusion, tol):
    print("\n[Test 2] DDIM eta=1 variance matches the DDPM posterior variance (adjacent steps, no model)")
    max_rel_err = 0.0
    for t in range(1, diffusion.scheduler.T, 37):  # sample every 37th t across the full range
        a_t = diffusion.scheduler.alphas_cumprod[t]
        a_prev = diffusion.scheduler.alphas_cumprod[t - 1]
        sigma_sq_ddim = (1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev)  # eta=1 -> eta^2=1
        sigma_sq_ddpm = diffusion.scheduler.posterior_variance[t]
        rel_err = ((sigma_sq_ddim - sigma_sq_ddpm).abs() / (sigma_sq_ddpm.abs() + 1e-12)).item()
        max_rel_err = max(max_rel_err, rel_err)

    ok = max_rel_err < tol
    print(f"  max relative error across sampled t in [1, {diffusion.scheduler.T}): {max_rel_err:.2e}  (tol={tol:.0e})")
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


# ===================================================================== #
# Test 3 (needs checkpoint): eta=0 DDIM sampling is deterministic.
# ===================================================================== #
def test_3_deterministic_given_seed(diffusion, num_classes, tol):
    print("\n[Test 3] eta=0 DDIM sampling is deterministic given the same x_T (needs checkpoint)")
    device = diffusion.device
    # DDIM eta=0 itself draws no reverse-process noise.  CUDA convolution
    # kernels can nevertheless be nondeterministic, and their tiny per-step
    # differences get amplified over 20 denoising steps.  Pin deterministic
    # kernels so this test measures DDIM determinism, not CUDA kernel choice.
    if device == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    shape = (4, 1, 32, 32)
    y = torch.randint(0, 10, (4,), device=device) if num_classes else None

    torch.manual_seed(123)
    x_T = torch.randn(shape, device=device)

    def run():
        x = x_T.clone()
        times = torch.linspace(diffusion.scheduler.T - 1, 0, steps=20).round().long().to(device)
        with torch.no_grad():
            for i in range(len(times)):
                t = int(times[i].item())
                t_prev = int(times[i + 1].item()) if i < len(times) - 1 else -1
                t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
                eps = diffusion._predict_noise(x, t_batch, y, guidance_scale=3.0 if num_classes else 1.0)
                x = diffusion.ddim_step(x, eps, t, t_prev, eta=0.0)
        return x

    out1 = run()
    out2 = run()
    max_err = (out1 - out2).abs().max().item()
    ok = max_err < tol
    print(f"  max abs difference between two eta=0 runs from the same x_T: {max_err:.2e}  (tol={tol:.0e})")
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


# ===================================================================== #
# Test 4 (needs checkpoint): sample quality should improve (error vs. a
#         high-step reference should shrink) as ddim_steps grows.
# ===================================================================== #
def test_4_quality_improves_with_more_steps(diffusion, num_classes):
    print("\n[Test 4] DDIM output converges toward the many-step reference as step count grows (needs checkpoint)")
    device = diffusion.device
    shape = (4, 1, 32, 32)
    y = torch.randint(0, 10, (4,), device=device) if num_classes else None
    guidance = 3.0 if num_classes else 1.0

    torch.manual_seed(7)
    x_T = torch.randn(shape, device=device)

    def run(steps):
        x = x_T.clone()
        times = torch.linspace(diffusion.scheduler.T - 1, 0, steps=steps).round().long().to(device)
        with torch.no_grad():
            for i in range(len(times)):
                t = int(times[i].item())
                t_prev = int(times[i + 1].item()) if i < len(times) - 1 else -1
                t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
                eps = diffusion._predict_noise(x, t_batch, y, guidance)
                x = diffusion.ddim_step(x, eps, t, t_prev, eta=0.0)
        return x

    reference = run(200)  # dense reference trajectory, same x_T
    errors = []
    for steps in [5, 10, 20, 50, 100]:
        out = run(steps)
        err = (out - reference).abs().mean().item()
        errors.append((steps, err))
        print(f"  steps={steps:<4d}  mean abs error vs 200-step reference = {err:.4f}")

    # Not a strict assert (network noise can cause small non-monotonicity),
    # just report whether the overall trend is decreasing.
    decreasing = errors[-1][1] < errors[0][1]
    print(f"  overall trend {'decreasing (expected)' if decreasing else 'NOT decreasing -- investigate'}")
    return decreasing


def build_model(args, device):
    scheduler = NoiseScheduler(timesteps=args.timesteps, schedule=args.schedule, device=device)
    num_classes = 10 if args.conditional else None
    model = UNet(in_channels=1, base_ch=args.base_ch, num_classes=num_classes).to(device)
    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
    model.eval()
    return GaussianDiffusion(model, scheduler, device), num_classes


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    diffusion, num_classes = build_model(args, device)

    results = {}
    results["test_1"] = test_1_algebraic_trajectory_consistency(diffusion, args.tol)
    results["test_2"] = test_2_eta1_matches_ddpm_posterior_variance(diffusion, args.tol)

    if args.ckpt is not None:
        results["test_3"] = test_3_deterministic_given_seed(diffusion, num_classes, args.tol)
        results["test_4"] = test_4_quality_improves_with_more_steps(diffusion, num_classes)
    else:
        print("\n(no --ckpt given, skipping Test 3 and Test 4 -- pass a checkpoint to also run those)")

    print("\n==================== SUMMARY ====================")
    for name, ok in results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    all_ok = all(results.values())
    print(f"\n{'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED -- see above'}")


if __name__ == "__main__":
    main()
