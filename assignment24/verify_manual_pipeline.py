"""
verify_manual_pipeline.py
==========================

Verification harness for `assignment_2_4_cfg_ddim.py`.

It does two kinds of checks:

1. STATIC checks (no GPU / no download needed):
   - Confirms `assignment_2_4_cfg_ddim.py` never calls a high-level pipeline
     (e.g. `StableDiffusionPipeline(...)` or `.from_pretrained(...)` on a
     pipeline class) anywhere in its own denoising path.
   - Confirms the CFG formula `uncond + scale * (cond - uncond)` appears in
     source (a literal correctness check on the implementation itself).

2. RUNTIME checks (needs the model + a GPU/CPU, downloads weights):
   - Runs the manual loop from `assignment_2_4_cfg_ddim.py` with a fixed
     seed/prompt/scale.
   - Runs the official `diffusers.StableDiffusionPipeline` with the SAME
     seed/prompt/scale/scheduler (DDIM) and the SAME number of steps.
   - Compares the two output images (MSE + PSNR). They won't be pixel-identical
     (fp16 kernel ordering / minor numeric differences), but they should be
     visually and numerically very close if the manual loop is implemented
     correctly. A large divergence signals a bug (wrong guidance formula,
     wrong scheduler config, embeddings swapped, etc).
   - Sanity-checks intermediate values: no NaNs/Infs in latents, output image
     is a valid 8-bit RGB array, latents have the expected shape.

Usage
-----
    # Static checks only (fast, no model download):
    python verify_manual_pipeline.py --static_only

    # Full verification (downloads the model, needs a GPU for reasonable speed):
    python verify_manual_pipeline.py \
        --prompt "a majestic lion wearing a crown, studio lighting" \
        --model_id runwayml/stable-diffusion-v1-5 \
        --steps 50 \
        --scale 7.5 \
        --seed 1234
"""

import argparse
import inspect
import sys

import numpy as np


# --------------------------------------------------------------------------- #
# 1. Static / source-level checks
# --------------------------------------------------------------------------- #
def run_static_checks(module_path: str = "assignment_2_4_cfg_ddim.py") -> bool:
    print("=== Static checks ===")
    ok = True

    with open(module_path, "r", encoding="utf-8") as f:
        source = f.read()

    # (a) no high-level pipeline call used to produce images
    forbidden = ["StableDiffusionPipeline(", "DiffusionPipeline.from_pretrained"]
    for token in forbidden:
        if token in source:
            print(f"  [FAIL] Found forbidden high-level pipeline usage: {token!r}")
            ok = False
    if ok:
        print("  [PASS] No high-level pipeline call found in assignment script")

    # (b) explicit CFG formula present
    cfg_signature_present = (
        "noise_pred_uncond + guidance_scale * (" in source
        or "noise_pred_uncond + scale * (" in source
    )
    if cfg_signature_present:
        print("  [PASS] Explicit CFG combination formula found in source")
    else:
        print("  [FAIL] Could not find explicit CFG formula (uncond + scale*(cond-uncond))")
        ok = False

    # (c) two separate UNet forward passes (cond + uncond), not a batched trick
    #     hidden from inspection — just check both calls exist textually.
    two_passes = source.count("unet(") >= 2
    if two_passes:
        print("  [PASS] At least two separate UNet forward-pass call sites found")
    else:
        print("  [FAIL] Expected two explicit unet(...) call sites (cond, uncond)")
        ok = False

    # (d) scheduler.step is called explicitly (manual DDIM stepping, not pipeline)
    if "scheduler.step(" in source:
        print("  [PASS] Explicit scheduler.step(...) call found")
    else:
        print("  [FAIL] No explicit scheduler.step(...) call found")
        ok = False

    print(f"=== Static checks {'PASSED' if ok else 'FAILED'} ===\n")
    return ok


# --------------------------------------------------------------------------- #
# 2. Runtime cross-check against the official pipeline
# --------------------------------------------------------------------------- #
def run_runtime_checks(prompt, model_id, steps, scale, seed, height=512, width=512):
    print("=== Runtime checks (this downloads/loads the model) ===")

    import torch
    from diffusers import StableDiffusionPipeline, DDIMScheduler

    # Import the manual implementation directly so we're testing the real code,
    # not a re-typed copy of it.
    sys.path.insert(0, ".")
    import assignment_2_4_cfg_ddim as manual

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"  Using device={device}, dtype={dtype}")

    # --- Manual pipeline (the code under test) ---
    tokenizer, text_encoder, vae, unet, scheduler = manual.load_components(
        model_id, device, dtype
    )
    cond_emb, uncond_emb = manual.get_cond_and_uncond_embeddings(
        prompt, tokenizer, text_encoder, device
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    latent_shape = (1, unet.config.in_channels, height // 8, width // 8)
    latents_init = torch.randn(latent_shape, generator=generator, device=device, dtype=dtype)

    manual_latents = manual.run_manual_cfg_ddim(
        unet=unet,
        scheduler=scheduler,
        cond_embeddings=cond_emb,
        uncond_embeddings=uncond_emb,
        guidance_scale=scale,
        num_inference_steps=steps,
        latents_init=latents_init,
        device=device,
        dtype=dtype,
    )

    # Sanity checks on intermediate values
    assert manual_latents.shape == latent_shape, (
        f"Unexpected latent shape: {manual_latents.shape} vs {latent_shape}"
    )
    assert torch.isfinite(manual_latents).all(), "Manual latents contain NaN/Inf"
    print("  [PASS] Manual latents: correct shape, no NaN/Inf")

    manual_image = manual.decode_latents(vae, manual_latents)
    manual_arr = np.array(manual_image).astype(np.float32)
    assert manual_arr.shape == (height, width, 3), "Decoded image has wrong shape"
    print("  [PASS] Manual pipeline produced a valid RGB image")

    # --- Official pipeline (reference / ground truth) ---
    ref_pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    ref_pipe.scheduler = DDIMScheduler.from_config(ref_pipe.scheduler.config)
    ref_pipe = ref_pipe.to(device)
    ref_pipe.set_progress_bar_config(disable=True)

    ref_generator = torch.Generator(device=device).manual_seed(seed)
    ref_latents_init = latents_init.clone()  # reuse identical starting noise

    ref_image = ref_pipe(
        prompt=prompt,
        negative_prompt="",
        guidance_scale=scale,
        num_inference_steps=steps,
        height=height,
        width=width,
        latents=ref_latents_init,
        generator=ref_generator,
    ).images[0]
    ref_arr = np.array(ref_image).astype(np.float32)

    # --- Compare ---
    mse = float(np.mean((manual_arr - ref_arr) ** 2))
    psnr = float("inf") if mse == 0 else 20 * np.log10(255.0) - 10 * np.log10(mse)
    print(f"  MSE  vs official pipeline: {mse:.4f}")
    print(f"  PSNR vs official pipeline: {psnr:.2f} dB")

    # PSNR > ~30 dB is a very close match for SD outputs with matched seeds/config;
    # lower than ~20 dB usually indicates a real implementation bug.
    passed = psnr > 25
    print(f"=== Runtime cross-check {'PASSED' if passed else 'FAILED'} (threshold: PSNR > 25 dB) ===")
    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--static_only", action="store_true")
    parser.add_argument("--prompt", type=str, default="a photo of an astronaut riding a horse")
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    static_ok = run_static_checks()

    if args.static_only:
        sys.exit(0 if static_ok else 1)

    runtime_ok = run_runtime_checks(
        prompt=args.prompt,
        model_id=args.model_id,
        steps=args.steps,
        scale=args.scale,
        seed=args.seed,
    )

    sys.exit(0 if (static_ok and runtime_ok) else 1)


if __name__ == "__main__":
    main()
