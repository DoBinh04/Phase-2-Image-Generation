"""
Assignment 2.7 - Inference cho Qwen-Image bang Diffusers, chay tren Vast.ai (RTX 3090/4090)

VRAM: RTX 3090 va 4090 deu co 24GB VRAM (4090 chi nhanh hon ve compute, khong nhieu
VRAM hon). Voi cach quantize 4-bit + offload duoi day, theo benchmark chinh thuc cua
Diffusers chi ton ~12.5GB -> 3090 la du, khong can len 4090.

Cach dung:
  1. Tao file .env cung thu muc voi noi dung:
       HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
  2. pip install -q -U diffusers transformers accelerate safetensors bitsandbytes python-dotenv huggingface_hub
  3. python qwen_image_vastai_inference.py --prompt "..." --steps 25
"""

import argparse
import os
import time

import torch
from dotenv import load_dotenv
from huggingface_hub import login
from diffusers import DiffusionPipeline
from diffusers.quantizers import PipelineQuantizationConfig

MODEL_ID = "Qwen/Qwen-Image"

DEFAULT_PROMPT = (
    "a tiny astronaut hatching from an egg on the moon, ultra HD, 4K, cinematic composition"
)


# ---------------------------------------------------------------------------
# MONKEYPATCH: vo hieu hoa `_caching_allocator_warmup`.
#
# Bug o tang thu vien (diffusers va transformers ban moi nhat): ham noi bo nay
# tinh SAI dung luong can "lam nong" GPU truoc khi load weight - tinh theo kich
# thuoc GOC (chua quantize) thay vi kich thuoc thuc te sau nen 4-bit, nen doi
# cap phat du thua rat nhieu va bi OOM du GPU du cho model. Bug nay xay ra tren
# nhieu loai GPU (ke ca GPU lon), khong rieng gi mot cau hinh nao.
# Ham nay chi la toi uu toc do load, tat di khong anh huong ket qua.
# ---------------------------------------------------------------------------
def _patch_caching_allocator_warmup():
    def _noop(*args, **kwargs):
        return None

    try:
        import diffusers.models.model_loading_utils as _dmlu
        _dmlu._caching_allocator_warmup = _noop
    except Exception as e:
        print(f"[patch] Bo qua patch diffusers: {e}")

    try:
        import transformers.modeling_utils as _tmu
        _tmu.caching_allocator_warmup = _noop
    except Exception as e:
        print(f"[patch] Bo qua patch transformers: {e}")


def setup_env():
    load_dotenv()  # doc file .env cung thu muc
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("[auth] Da dang nhap Hugging Face bang HF_TOKEN tu .env")
    else:
        print("[auth] Khong thay HF_TOKEN trong .env - se tai model o che do public/anonymous")

    if not torch.cuda.is_available():
        raise RuntimeError("Khong tim thay GPU. Kiem tra lai instance Vast.ai da gan GPU chua.")
    name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"[gpu] {name} - {vram_gb:.1f} GB VRAM")

    _patch_caching_allocator_warmup()


def load_pipeline(use_quantization: bool = True, use_offload: bool = True) -> DiffusionPipeline:
    """
    RTX 3090/4090 (Ampere/Ada) ho tro bfloat16 native tensor core -> dung bf16
    thay vi fp16 (khac voi T4 doi Turing khong co bf16 tensor core).
    """
    compute_dtype = torch.bfloat16

    if use_quantization:
        quant_config = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": compute_dtype,
            },
            components_to_quantize=["transformer", "text_encoder"],
        )
        print("[load] Dang tai + quantize Qwen-Image (4-bit NF4)...")
        pipe = DiffusionPipeline.from_pretrained(
            MODEL_ID,
            dtype=compute_dtype,
            quantization_config=quant_config,
            device_map="cuda",
        )
        if use_offload:
            # Offload de them headroom cho activations; voi 24GB VRAM co the bo
            # dong nay de toc do cao nhat (--no-offload).
            pipe.enable_model_cpu_offload()
    else:
        # 24GB co the khong du cho ban FULL bf16 (~45GB) -> chi dung khi ban da
        # co quantized checkpoint rieng hoac muon test tren nhieu GPU (multi-GPU).
        print("[load] Dang tai Qwen-Image (bf16, KHONG quantize)...")
        pipe = DiffusionPipeline.from_pretrained(MODEL_ID, dtype=compute_dtype, device_map="cuda")

    return pipe


def generate(
    pipe: DiffusionPipeline,
    prompt: str,
    negative_prompt: str = " ",
    steps: int = 25,
    true_cfg_scale: float = 4.0,
    seed: int = 0,
    width: int = 1024,
    height: int = 1024,
):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,   # can co (ke ca rong) de bat true CFG
        width=width,
        height=height,
        num_inference_steps=steps,
        true_cfg_scale=true_cfg_scale,
        generator=generator,
    ).images[0]
    return image


def main():
    parser = argparse.ArgumentParser(description="Qwen-Image inference tren Vast.ai (RTX 3090/4090)")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", type=str, default=" ")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--cfg", type=float, default=4.0, dest="true_cfg_scale")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--out", type=str, default="qwen_image_output.png")
    parser.add_argument("--no-quantize", action="store_true",
                         help="Tat quantization (can >=40GB VRAM, KHONG khuyen nghi tren 1x 3090/4090)")
    parser.add_argument("--no-offload", action="store_true",
                         help="Tat CPU offload - toc do cao hon, chi nen dung khi thua VRAM ro rang")
    args = parser.parse_args()

    setup_env()
    pipe = load_pipeline(use_quantization=not args.no_quantize, use_offload=not args.no_offload)

    print("[run] Bat dau sinh anh...")
    t0 = time.time()
    image = generate(
        pipe,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        steps=args.steps,
        true_cfg_scale=args.true_cfg_scale,
        seed=args.seed,
        width=args.width,
        height=args.height,
    )
    print(f"[run] Xong sau {time.time() - t0:.1f}s")

    image.save(args.out)
    print(f"[run] Da luu anh: {args.out}")


if __name__ == "__main__":
    main()
