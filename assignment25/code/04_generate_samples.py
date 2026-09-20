"""
04_generate_samples.py
Sinh anh test bang base SD1.5 + LoRA da train (sks_dog_lora).
Dung prompt KHAC voi caption luc train de danh gia khach quan
kha nang generalize (khong chi hoc thuoc long anh train).

Chay:
    source /workspace/sd-scripts/venv/bin/activate
    python3 04_generate_samples.py
"""

import os
import json
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

WS = "/workspace"
BASE_MODEL = os.path.join(WS, "models", "v1-5-pruned-emaonly-fp16.safetensors")
LORA_PATH = os.path.join(WS, "lora_train", "output", "sks_dog_lora.safetensors")
OUTPUT_DIR = os.path.join(WS, "lora_train", "eval", "generated")
LORA_STRENGTH = float(os.environ.get("LORA_STRENGTH", 0.8))

# Prompt test - CO CHU Y khac voi caption luc train (test generalization)
TEST_PROMPTS = [
    "a photo of sks dog running on the beach",
    "a photo of sks dog wearing a red hat",
    "a painting of sks dog in the style of van gogh",
    "a photo of sks dog sleeping on a sofa",
    "a photo of sks dog in the snow",
    "a photo of sks dog playing with a ball",
    "a photo of sks dog in a garden full of flowers",
    "a professional studio photo of sks dog",
    "a photo of sks dog swimming in a lake",
    "a cartoon illustration of sks dog",
]

NEGATIVE_PROMPT = "blurry, low quality, deformed, extra limbs, bad anatomy, watermark"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    assert os.path.exists(LORA_PATH), (
        f"Khong tim thay LoRA: {LORA_PATH}. Chay ./03_train_lora.sh truoc."
    )

    print(">>> Loading base SD1.5 pipeline...")
    pipe = StableDiffusionPipeline.from_single_file(
        BASE_MODEL,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    ).to("cuda")

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True
    )

    print(f">>> Loading LoRA weights: {LORA_PATH} (scale={LORA_STRENGTH})")
    pipe.load_lora_weights(LORA_PATH)
    pipe.fuse_lora(lora_scale=LORA_STRENGTH)

    # PyTorch SDPA da bat san trong diffusers moi -> KHONG goi xformers
    # (ban xformers tren PyPI khong khop torch 2.5.1 se lam hong pipeline).
    pipe.set_progress_bar_config(disable=True)

    metadata = []
    for idx, prompt in enumerate(TEST_PROMPTS):
        print(f">>> [{idx+1}/{len(TEST_PROMPTS)}] Generating: {prompt}")
        generator = torch.Generator(device="cuda").manual_seed(1000 + idx)
        image = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=25,
            guidance_scale=7.0,
            width=512,
            height=512,
            generator=generator,
        ).images[0]

        filename = f"gen_{idx:02d}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        image.save(filepath)
        metadata.append({"file": filename, "prompt": prompt})

    # Luu metadata (prompt <-> anh) de script danh gia CLIP-T doc lai
    meta_path = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(
            {"lora_strength": LORA_STRENGTH, "items": metadata},
            f, indent=2, ensure_ascii=False,
        )

    print("")
    print("==========================================")
    print(f"  DA SINH {len(TEST_PROMPTS)} ANH TEST TAI: {OUTPUT_DIR}")
    print(f"  Metadata: {meta_path}")
    print("  Buoc tiep theo: python3 05_evaluate_clip.py")
    print("==========================================")


if __name__ == "__main__":
    main()
