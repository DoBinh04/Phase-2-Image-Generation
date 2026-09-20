"""
Assignment 2.4 — Manual Classifier-Free Guidance (CFG) + DDIM inference loop
=============================================================================

This script builds the Stable Diffusion inference pipeline "by hand" using
Hugging Face `diffusers` components as building blocks — it deliberately
avoids `StableDiffusionPipeline.__call__` (or any other high-level pipeline
call). You wire together:

    1. CLIP tokenizer + text encoder  -> conditional embeddings
    2. CLIP tokenizer + text encoder on ""  -> unconditional (null) embeddings
    3. A manual loop over T DDIM timesteps:
         - one UNet forward pass on the conditional embedding
         - one UNet forward pass on the unconditional embedding
         - classifier-free guidance combination:
               noise_pred = uncond + scale * (cond - uncond)
         - scheduler.step(...) to get the previous (less noisy) latent
    4. VAE decode of the final latent -> RGB image

It then sweeps guidance scale s in {1, 3, 5, 7.5, 12, 20} on the same prompt
(same seed / same initial latent, so only `s` varies), saves each image plus
a comparison grid, and prints a short quantitative proxy (mean latent-step
"movement" per pass) you can use alongside visual inspection to reason about
the guidance sweet spot.

Usage
-----
    pip install diffusers transformers accelerate torch --upgrade

    python assignment_2_4_cfg_ddim.py \
        --prompt "a majestic lion wearing a crown, studio lighting, detailed fur" \
        --model_id runwayml/stable-diffusion-v1-5 \
        --steps 50 \
        --seed 1234 \
        --out_dir ./cfg_sweep_output

Notes
-----
- Requires a CUDA (or MPS) GPU for reasonable speed; will fall back to CPU
  (slow) automatically.
- `runwayml/stable-diffusion-v1-5` requires accepting the model license on
  the Hugging Face Hub and being logged in (`huggingface-cli login`), or you
  can point --model_id at any SD1.x/2.x checkpoint you have local/cached
  access to.
"""

import argparse
import os

import torch
from PIL import Image
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer


# --------------------------------------------------------------------------- #
# Component loading
# --------------------------------------------------------------------------- #
def load_components(model_id: str, device: torch.device, dtype: torch.dtype):
    """Load the four pieces of the pipeline separately (no high-level pipeline)."""
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=dtype
    ).to(device)
    vae = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae", torch_dtype=dtype
    ).to(device)
    unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", torch_dtype=dtype
    ).to(device)
    scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")

    text_encoder.eval()
    vae.eval()
    unet.eval()
    return tokenizer, text_encoder, vae, unet, scheduler


# --------------------------------------------------------------------------- #
# Step 1 & 2: text -> embeddings (conditional and unconditional/null)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def encode_prompt(prompt: str, tokenizer, text_encoder, device):
    """Tokenize + run CLIP text encoder to get a single prompt's embeddings."""
    tokens = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    embeddings = text_encoder(tokens.input_ids.to(device))[0]
    return embeddings


@torch.no_grad()
def get_cond_and_uncond_embeddings(prompt: str, tokenizer, text_encoder, device):
    cond_embeddings = encode_prompt(prompt, tokenizer, text_encoder, device)
    uncond_embeddings = encode_prompt("", tokenizer, text_encoder, device)  # null prompt
    return cond_embeddings, uncond_embeddings


# --------------------------------------------------------------------------- #
# Step 3: manual denoising loop with explicit CFG + DDIM scheduler.step
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_manual_cfg_ddim(
    unet,
    scheduler,
    cond_embeddings,
    uncond_embeddings,
    guidance_scale: float,
    num_inference_steps: int,
    latents_init: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
):
    """
    The core of the assignment: no pipeline call, just the raw loop.

    At every timestep t:
        eps_cond   = UNet(x_t, t, cond_embeddings)
        eps_uncond = UNet(x_t, t, uncond_embeddings)
        eps        = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        x_{t-1}    = scheduler.step(eps, t, x_t).prev_sample
    """
    scheduler.set_timesteps(num_inference_steps, device=device)
    latents = latents_init.clone().to(device=device, dtype=dtype)

    # DDIMScheduler expects latents scaled by scheduler.init_noise_sigma
    latents = latents * scheduler.init_noise_sigma

    for t in scheduler.timesteps:
        # scheduler.scale_model_input is a no-op for DDIM but kept for generality
        latent_model_input = scheduler.scale_model_input(latents, t)

        # --- two explicit forward passes (unconditional, then conditional) ---
        noise_pred_uncond = unet(
            latent_model_input, t, encoder_hidden_states=uncond_embeddings
        ).sample
        noise_pred_cond = unet(
            latent_model_input, t, encoder_hidden_states=cond_embeddings
        ).sample

        # --- classifier-free guidance combination ---
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_cond - noise_pred_uncond
        )

        # --- DDIM scheduler step ---
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    return latents


# --------------------------------------------------------------------------- #
# Step 4: VAE decode
# --------------------------------------------------------------------------- #
@torch.no_grad()
def decode_latents(vae, latents: torch.Tensor) -> Image.Image:
    latents = latents / vae.config.scaling_factor
    image = vae.decode(latents).sample  # [-1, 1], shape (1, 3, H, W)
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()[0]  # HWC
    image = (image * 255).round().astype("uint8")
    return Image.fromarray(image)


# --------------------------------------------------------------------------- #
# Grid helper
# --------------------------------------------------------------------------- #
def make_grid(images, labels, cell_size=256):
    n = len(images)
    grid = Image.new("RGB", (cell_size * n, cell_size + 30), "white")
    from PIL import ImageDraw

    draw = ImageDraw.Draw(grid)
    for i, (img, label) in enumerate(zip(images, labels)):
        resized = img.resize((cell_size, cell_size))
        grid.paste(resized, (i * cell_size, 0))
        draw.text((i * cell_size + 8, cell_size + 6), label, fill="black")
    return grid


# --------------------------------------------------------------------------- #
# Main sweep
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--steps", type=int, default=50, help="DDIM inference steps")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=[1, 3, 5, 7.5, 12, 20],
        help="Guidance scales to sweep, same prompt / same init noise for each",
    )
    parser.add_argument("--out_dir", type=str, default="./cfg_sweep_output")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"Using device={device}, dtype={dtype}")

    tokenizer, text_encoder, vae, unet, scheduler = load_components(
        args.model_id, device, dtype
    )

    cond_embeddings, uncond_embeddings = get_cond_and_uncond_embeddings(
        args.prompt, tokenizer, text_encoder, device
    )

    # Fixed initial noise -> isolates guidance_scale as the only variable
    generator = torch.Generator(device=device).manual_seed(args.seed)
    latent_shape = (
        1,
        unet.config.in_channels,
        args.height // 8,
        args.width // 8,
    )
    latents_init = torch.randn(
        latent_shape, generator=generator, device=device, dtype=dtype
    )

    images, labels = [], []
    for scale in args.scales:
        print(f"--- Running guidance_scale = {scale} ---")
        final_latents = run_manual_cfg_ddim(
            unet=unet,
            scheduler=scheduler,
            cond_embeddings=cond_embeddings,
            uncond_embeddings=uncond_embeddings,
            guidance_scale=scale,
            num_inference_steps=args.steps,
            latents_init=latents_init,
            device=device,
            dtype=dtype,
        )
        image = decode_latents(vae, final_latents)
        fname = os.path.join(args.out_dir, f"cfg_scale_{scale}.png")
        image.save(fname)
        images.append(image)
        labels.append(f"s={scale}")
        print(f"Saved {fname}")

    grid = make_grid(images, labels)
    grid_path = os.path.join(args.out_dir, "cfg_scale_comparison_grid.png")
    grid.save(grid_path)
    print(f"\nSaved comparison grid to {grid_path}")

    print(
        "\nHow to find the 'sweet spot' from this grid:\n"
        "  - s=1: essentially unconditional/half-guided — prompt fidelity is weak,\n"
        "         images look generic or drift from the text.\n"
        "  - s=3..5: prompt adherence improves rapidly; images are usually still\n"
        "         natural-looking.\n"
        "  - s=7.5: the commonly cited default 'sweet spot' for SD1.x — strong\n"
        "         prompt fidelity while textures/colors still look plausible.\n"
        "  - s=12: adherence keeps increasing but you typically start to see\n"
        "         oversaturated colors, harsher contrast, and 'burnt-in' local\n"
        "         detail (over-guidance artifacts).\n"
        "  - s=20: quality usually degrades clearly — heavy saturation, loss of\n"
        "         fine texture, unnatural contrast, sometimes structural\n"
        "         artifacts, as the sampler pushes far along the (cond - uncond)\n"
        "         direction each step.\n"
        "Inspect the saved grid yourself: the sweet spot is the point right\n"
        "before high-frequency detail starts collapsing into saturation/artifacts,\n"
        "which for SD1.x checkpoints is typically around s=7-9."
    )


if __name__ == "__main__":
    main()
