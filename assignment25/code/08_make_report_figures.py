#!/usr/bin/env python3
"""
08_make_report_figures.py
Sinh cac hinh minh hoa cho bao cao (README.md), luu vao /workspace/report_images/:

  fig1_dataset.jpg        - 5 anh training + caption
  fig2_generated.jpg      - 10 anh sinh ra tu 10 prompt moi (LoRA strength 0.8)
  fig3_ablation.jpg       - cung prompt/seed o 3 muc LoRA strength 0.6 / 0.8 / 1.0
  fig4_comfy_compare.jpg  - ComfyUI: baseline vs fast vs full (CUNG seed, cung prompt)
  fig5_hiresfix.jpg       - Tai sao can hires-fix: sinh thang 1536px vs hires-fix

Chay:
    source /workspace/sd-scripts/venv/bin/activate
    python3 08_make_report_figures.py
"""
import json
import os
import textwrap
import time
import urllib.request
import uuid

from PIL import Image, ImageDraw, ImageFont

WS = "/workspace"
OUT = os.path.join(WS, "report_images")
EVAL = os.path.join(WS, "lora_train", "eval")
TRAIN_DIR = os.path.join(WS, "lora_train", "img", "10_sks_dog")
COMFY = "http://127.0.0.1:18188"
COMFY_OUT = os.path.join(WS, "ComfyUI", "output")
API_WF = os.path.join(WS, "workflows", "sks_dog_inference_api.json")

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

BENCH_PROMPT = ("a professional photo of sks dog sitting in a garden full of flowers, "
                "golden hour sunlight, shallow depth of field, highly detailed fur, sharp focus")
BENCH_NEG = "blurry, low quality, deformed, extra limbs, bad anatomy, watermark, text"
BENCH_SEED = 777777


def font(size, bold=False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT_PATH, size)


def grid(items, cols, cell=320, caption_h=70, title=None, title_h=56, wrap=42):
    """items: list of (PIL.Image, caption). Tra ve mot anh luoi co chu thich."""
    rows = (len(items) + cols - 1) // cols
    pad = 12
    W = cols * cell + pad * (cols + 1)
    top = title_h if title else 0
    H = top + rows * (cell + caption_h) + pad * (rows + 1)
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)

    if title:
        d.text((pad, pad + 4), title, fill="black", font=font(26, bold=True))

    for i, (img, cap) in enumerate(items):
        r, c = divmod(i, cols)
        x = pad + c * (cell + pad)
        y = top + pad + r * (cell + caption_h + pad)
        canvas.paste(img.convert("RGB").resize((cell, cell), Image.LANCZOS), (x, y))
        lines = textwrap.wrap(cap, wrap)[:3]
        for j, line in enumerate(lines):
            d.text((x, y + cell + 4 + j * 18), line, fill="black", font=font(14))
    return canvas


# ---------- ComfyUI helpers ----------
def _post(path, payload):
    req = urllib.request.Request(COMFY + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def _get(path):
    return json.loads(urllib.request.urlopen(COMFY + path, timeout=60).read())


def comfy_run(graph):
    t0 = time.time()
    pid = _post("/prompt", {"prompt": graph, "client_id": str(uuid.uuid4())})["prompt_id"]
    while True:
        h = _get(f"/history/{pid}")
        if pid in h:
            st = h[pid].get("status", {})
            if st.get("completed"):
                break
            if st.get("status_str") == "error":
                raise RuntimeError(json.dumps(st)[:2000])
        time.sleep(0.2)
    files = [i["filename"] for o in h[pid]["outputs"].values() for i in o.get("images", [])]
    return time.time() - t0, files[-1]


def graph_baseline(seed, prefix):
    return {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": "v1-5-pruned-emaonly-fp16.safetensors"}},
        "2": {"class_type": "LoraLoader",
              "inputs": {"model": ["1", 0], "clip": ["1", 1],
                         "lora_name": "sks_dog_lora.safetensors",
                         "strength_model": 0.75, "strength_clip": 0.75}},
        "20": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 1], "text": BENCH_PROMPT}},
        "21": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 1], "text": BENCH_NEG}},
        "22": {"class_type": "EmptyLatentImage",
               "inputs": {"width": 512, "height": 512, "batch_size": 1}},
        "23": {"class_type": "KSampler",
               "inputs": {"model": ["2", 0], "positive": ["20", 0], "negative": ["21", 0],
                          "latent_image": ["22", 0], "seed": seed, "steps": 30, "cfg": 7.5,
                          "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0}},
        "24": {"class_type": "VAEDecode", "inputs": {"samples": ["23", 0], "vae": ["1", 2]}},
        "25": {"class_type": "SaveImage",
               "inputs": {"images": ["24", 0], "filename_prefix": prefix}},
    }


def graph_fast(seed, prefix):
    g = graph_baseline(seed, prefix)
    g["3"] = {"class_type": "LoraLoader",
              "inputs": {"model": ["2", 0], "clip": ["2", 1],
                         "lora_name": "lcm-lora-sdv1-5.safetensors",
                         "strength_model": 1.0, "strength_clip": 1.0}}
    g["4"] = {"class_type": "FreeU_V2",
              "inputs": {"model": ["3", 0], "b1": 1.3, "b2": 1.4, "s1": 0.9, "s2": 0.2}}
    g["20"]["inputs"]["clip"] = ["3", 1]
    g["21"]["inputs"]["clip"] = ["3", 1]
    g["23"]["inputs"].update({"model": ["4", 0], "steps": 8, "cfg": 2.0,
                              "sampler_name": "lcm", "scheduler": "sgm_uniform"})
    return g


def graph_full(seed, prefix):
    with open(API_WF) as f:
        g = json.load(f)
    g["6"]["inputs"]["text"] = BENCH_PROMPT
    g["7"]["inputs"]["text"] = BENCH_NEG
    g["9"]["inputs"]["seed"] = seed
    g["11"]["inputs"]["seed"] = seed
    g["16"]["inputs"]["filename_prefix"] = prefix
    return g


def fig_dataset():
    items = []
    for p in sorted(f for f in os.listdir(TRAIN_DIR) if f.endswith(".jpg")):
        cap_path = os.path.join(TRAIN_DIR, os.path.splitext(p)[0] + ".txt")
        cap = open(cap_path).read().strip() if os.path.exists(cap_path) else ""
        items.append((Image.open(os.path.join(TRAIN_DIR, p)), f"{p} — {cap}"))
    return grid(items, cols=5, cell=300, caption_h=76,
                title="Hinh 1. Bo du lieu training (5 anh, subject 'sks dog') + caption", wrap=40)


def fig_generated():
    meta = json.load(open(os.path.join(EVAL, "generated", "metadata.json")))
    items = []
    for it in meta["items"]:
        img = Image.open(os.path.join(EVAL, "generated", it["file"]))
        items.append((img, it["prompt"]))
    return grid(items, cols=5, cell=300, caption_h=60,
                title="Hinh 2. 10 anh sinh tu 10 prompt MOI (LoRA strength 0.8) "
                      "- CLIP-I 0.8837 / CLIP-T 0.2813", wrap=38)


def fig_ablation():
    ab = json.load(open(os.path.join(EVAL, "ablation_clip.json")))
    meta = json.load(open(os.path.join(EVAL, "generated", "metadata.json")))
    picks = ["gen_01.png", "gen_02.png", "gen_06.png"]  # red hat / van gogh / garden
    prompts = {it["file"]: it["prompt"] for it in meta["items"]}
    items = []
    for row in ab:
        s = row["lora_strength"]
        d = os.path.join(EVAL, f"generated_s{str(s).replace('.', '')}")
        for fn in picks:
            cap = (f"strength {s} | CLIP-I {row['CLIP-I']:.4f} CLIP-T {row['CLIP-T']:.4f}\n"
                   + prompts[fn])
            items.append((Image.open(os.path.join(d, fn)), cap.replace("\n", " — ")))
    return grid(items, cols=3, cell=330, caption_h=76,
                title="Hinh 3. Anh huong cua LoRA strength (cung prompt, cung seed)", wrap=46)


def fig_comfy():
    runs = [
        ("A. Baseline: euler 30 steps, CFG 7.5, 512px", graph_baseline, "fig_A"),
        ("B. + LCM-LoRA + FreeU_V2: 8 steps, CFG 2.0, 512px", graph_fast, "fig_B"),
        ("C. Full: B + hires-fix 768 + ESRGAN 4x, 1536px", graph_full, "fig_C"),
    ]
    items = []
    for label, fn, prefix in runs:
        # warm-up: nap model vao VRAM truoc (seed khac de khong dinh cache cua ComfyUI),
        # neu khong thi con so do duoc se gom ca thoi gian load model -> lech voi
        # bang benchmark o 06_comfy_benchmark.py
        comfy_run(fn(BENCH_SEED + 1, prefix + "_warmup"))
        secs, filename = comfy_run(fn(BENCH_SEED, prefix))
        img = Image.open(os.path.join(COMFY_OUT, filename))
        items.append((img, f"{label} — {secs:.2f}s — {img.width}x{img.height}px"))
        print(f"  {label}: {secs:.2f}s -> {filename} ({img.width}x{img.height})")
    return grid(items, cols=3, cell=420, caption_h=64,
                title="Hinh 4. ComfyUI: baseline vs toi uu (CUNG seed, CUNG prompt)", wrap=54)


def graph_native_1536(seed, prefix):
    """Baseline sinh THANG o 1536px - de chung minh vi sao can hires-fix."""
    g = graph_baseline(seed, prefix)
    g["22"]["inputs"].update({"width": 1536, "height": 1536})
    return g


def fig_hiresfix():
    """So sanh sinh thang 1536px (hong bo cuc) vs hires-fix 512->768->ESRGAN."""
    items = []
    for label, fn, prefix in [
        ("A. Baseline sinh THANG o 1536px (30 steps)", graph_native_1536, "fig5_native"),
        ("C. Hires-fix: 512 -> 768 -> ESRGAN 4x -> 1536px", graph_full, "fig5_hires"),
    ]:
        comfy_run(fn(BENCH_SEED + 1, prefix + "_warmup"))
        secs, filename = comfy_run(fn(BENCH_SEED, prefix))
        img = Image.open(os.path.join(COMFY_OUT, filename))
        items.append((img, f"{label} — {secs:.2f}s — {img.width}x{img.height}px"))
        print(f"  {label}: {secs:.2f}s -> {filename}")
    return grid(items, cols=2, cell=560, caption_h=64,
                title="Hinh 5. Vi sao can hires-fix: SD1.5 sinh thang o do phan giai cao "
                      "bi lap chu the", wrap=72)


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, builder in [("fig1_dataset", fig_dataset),
                          ("fig2_generated", fig_generated),
                          ("fig3_ablation", fig_ablation),
                          ("fig4_comfy_compare", fig_comfy),
                          ("fig5_hiresfix", fig_hiresfix)]:
        print(f">>> {name}")
        img = builder()
        path = os.path.join(OUT, name + ".jpg")
        img.save(path, quality=88)
        print(f"    -> {path} ({img.width}x{img.height})")


if __name__ == "__main__":
    main()
