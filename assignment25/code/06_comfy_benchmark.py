#!/usr/bin/env python3
"""
06_comfy_benchmark.py
Do TOC DO + xuat anh cua cac workflow ComfyUI de chung minh phan
"improve image quality and inference speed" cua de bai:

  A. BASELINE  : SD1.5 + sks_dog_lora, KSampler euler 30 steps, CFG 7.5, 512px
  B. FAST      : + LCM-LoRA, sampler "lcm", 8 steps, CFG 1.5, 512px
                 -> so sanh toc do CONG BANG voi A (cung 512px, cung 1 anh)
  C. FULL      : B + FreeU_V2 + hires-fix 768 + ESRGAN 4x-UltraSharp -> 1536px
                 -> chat luong cao nhat, van nhanh hon A du anh to gap 3

LUU Y KY THUAT: ComfyUI CACHE ket qua theo hash input. Neu chay lai cung
seed thi no tra ve ngay lap tuc (~0.2s) va so do vo nghia -> moi lan chay
phai dung seed khac nhau.

Chay:
    python3 06_comfy_benchmark.py
Ket qua: /workspace/lora_train/eval/comfy_benchmark.json
         anh nam trong /workspace/ComfyUI/output/
"""
import json
import time
import urllib.request
import uuid
import os
import random
import statistics

COMFY = "http://127.0.0.1:18188"
API_WF = "/workspace/workflows/sks_dog_inference_api.json"
RESULT = "/workspace/lora_train/eval/comfy_benchmark.json"
REPEATS = 3

PROMPT = ("a professional photo of sks dog sitting in a garden full of flowers, "
          "golden hour sunlight, shallow depth of field, highly detailed fur, sharp focus")
NEGATIVE = "blurry, low quality, deformed, extra limbs, bad anatomy, watermark, text"


def post(path, payload):
    req = urllib.request.Request(
        COMFY + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        return json.loads(urllib.request.urlopen(req, timeout=60).read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"ComfyUI tu choi graph ({e.code}):\n"
                           + e.read().decode()[:3000]) from None


def get(path):
    return json.loads(urllib.request.urlopen(COMFY + path, timeout=60).read())


def run(graph):
    t0 = time.time()
    pid = post("/prompt", {"prompt": graph, "client_id": str(uuid.uuid4())})["prompt_id"]
    while True:
        hist = get(f"/history/{pid}")
        if pid in hist:
            status = hist[pid].get("status", {})
            if status.get("completed"):
                break
            if status.get("status_str") == "error":
                raise RuntimeError(json.dumps(hist[pid]["status"], indent=2)[:3000])
        time.sleep(0.2)
    elapsed = time.time() - t0
    files = [img["filename"]
             for out in hist[pid]["outputs"].values()
             for img in out.get("images", [])]
    return elapsed, files


def _common(prefix, seed, lora_nodes, model_out, clip_src, steps, cfg, sampler, scheduler):
    """Khung chung cho graph baseline/fast (512px, 1 pass)."""
    g = {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": "v1-5-pruned-emaonly-fp16.safetensors"}},
    }
    g.update(lora_nodes)
    g.update({
        "20": {"class_type": "CLIPTextEncode",
               "inputs": {"clip": [clip_src, 1], "text": PROMPT}},
        "21": {"class_type": "CLIPTextEncode",
               "inputs": {"clip": [clip_src, 1], "text": NEGATIVE}},
        "22": {"class_type": "EmptyLatentImage",
               "inputs": {"width": 512, "height": 512, "batch_size": 1}},
        "23": {"class_type": "KSampler",
               "inputs": {"model": model_out, "positive": ["20", 0], "negative": ["21", 0],
                          "latent_image": ["22", 0], "seed": seed, "steps": steps,
                          "cfg": cfg, "sampler_name": sampler,
                          "scheduler": scheduler, "denoise": 1.0}},
        "24": {"class_type": "VAEDecode", "inputs": {"samples": ["23", 0], "vae": ["1", 2]}},
        "25": {"class_type": "SaveImage",
               "inputs": {"images": ["24", 0], "filename_prefix": prefix}},
    })
    return g


def graph_a(seed):
    """Baseline: chi LoRA subject, euler 30 steps."""
    lora = {"2": {"class_type": "LoraLoader",
                  "inputs": {"model": ["1", 0], "clip": ["1", 1],
                             "lora_name": "sks_dog_lora.safetensors",
                             "strength_model": 0.75, "strength_clip": 0.75}}}
    return _common("bench_A_baseline", seed, lora, ["2", 0], "2", 30, 7.5, "euler", "normal")


def graph_b(seed):
    """Fast: + LCM-LoRA, sampler lcm, 8 steps.

    KHONG dung ModelSamplingDiscrete(lcm): tren ComfyUI 0.35 no cong don voi
    sampler "lcm" va lam anh ra nhieu trang.
    """
    lora = {
        "2": {"class_type": "LoraLoader",
              "inputs": {"model": ["1", 0], "clip": ["1", 1],
                         "lora_name": "sks_dog_lora.safetensors",
                         "strength_model": 0.75, "strength_clip": 0.75}},
        "3": {"class_type": "LoraLoader",
              "inputs": {"model": ["2", 0], "clip": ["2", 1],
                         "lora_name": "lcm-lora-sdv1-5.safetensors",
                         "strength_model": 1.0, "strength_clip": 1.0}},
        "4": {"class_type": "FreeU_V2",
              "inputs": {"model": ["3", 0], "b1": 1.3, "b2": 1.4, "s1": 0.9, "s2": 0.2}},
    }
    # CLIP lay tu node 3 (LoraLoader cuoi), KHONG phai node 4
    # (FreeU_V2 chi xuat MODEL, khong co CLIP).
    return _common("bench_B_fast", seed, lora, ["4", 0], "3", 8, 2.0, "lcm", "sgm_uniform")


def graph_c(seed):
    """Full: workflow toi uu day du (file sks_dog_inference_api.json)."""
    with open(API_WF) as f:
        g = json.load(f)
    g["6"]["inputs"]["text"] = PROMPT
    g["7"]["inputs"]["text"] = NEGATIVE
    g["9"]["inputs"]["seed"] = seed
    g["11"]["inputs"]["seed"] = seed
    g["16"]["inputs"]["filename_prefix"] = "bench_C_full"
    return g


VARIANTS = [
    ("A_baseline_euler30_512px", graph_a),
    ("B_fast_lcm8_freeu_512px", graph_b),
    ("C_full_lcm8_freeu_hires_upscale_1536px", graph_c),
]


def main():
    print(">>> Warm-up (nap model vao VRAM, khong tinh gio)...")
    for _, fn in VARIANTS:
        run(fn(random.randint(1, 10**8)))

    results = []
    for label, fn in VARIANTS:
        times, files = [], []
        for i in range(REPEATS):
            # seed khac nhau moi lan -> tranh cache cua ComfyUI
            secs, f = run(fn(random.randint(1, 10**8)))
            times.append(secs)
            files.extend(f)
            print(f"  {label} [{i+1}/{REPEATS}]: {secs:.2f}s")
        results.append({
            "label": label,
            "runs_sec": [round(t, 2) for t in times],
            "median_sec": round(statistics.median(times), 2),
            "images": files,
        })

    base = results[0]["median_sec"]
    for r in results:
        r["speedup_vs_baseline"] = round(base / r["median_sec"], 2)

    os.makedirs(os.path.dirname(RESULT), exist_ok=True)
    with open(RESULT, "w") as f:
        json.dump({"repeats": REPEATS, "results": results}, f, indent=2)

    print("")
    print("==========================================")
    for r in results:
        print(f"  {r['label']:42s} {r['median_sec']:6.2f}s  ({r['speedup_vs_baseline']}x)")
    print(f"  Chi tiet: {RESULT}")
    print("  Anh: /workspace/ComfyUI/output/")
    print("==========================================")


if __name__ == "__main__":
    main()
