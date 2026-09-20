"""
05_evaluate_clip.py
Danh gia LoRA da train bang 2 metric (theo DreamBooth, Ruiz et al. 2023):
  - CLIP-I: subject fidelity = cosine similarity trung binh giua embedding
            anh THAT (training set) va anh GENERATE
  - CLIP-T: prompt fidelity  = cosine similarity giua embedding prompt
            va embedding anh generate tuong ung

Chay:
    source /workspace/sd-scripts/venv/bin/activate
    python3 05_evaluate_clip.py
"""

import os
import json
import glob
import torch
import open_clip
from PIL import Image

WS = "/workspace"
REAL_IMG_DIR = os.path.join(WS, "lora_train", "img", "10_sks_dog")
GEN_IMG_DIR = os.path.join(WS, "lora_train", "eval", "generated")
META_PATH = os.path.join(GEN_IMG_DIR, "metadata.json")
RESULT_PATH = os.path.join(WS, "lora_train", "eval", "clip_scores.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-B-32-quickgelu"


def load_model():
    # LUU Y: weights "openai" duoc train voi QuickGELU. Tu open_clip 2.24+
    # phai goi ten model la "ViT-B-32-quickgelu", neu dung "ViT-B-32" thi
    # activation bi sai (GELU) -> diem CLIP lech. Day la loi rat de bo sot.
    print(">>> Loading CLIP ViT-B-32-quickgelu (open_clip, openai weights)...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    model.eval().to(DEVICE)
    return model, preprocess, tokenizer


@torch.no_grad()
def embed_image(model, preprocess, path):
    img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)
    feat = model.encode_image(img)
    return feat / feat.norm(dim=-1, keepdim=True)


@torch.no_grad()
def embed_text(model, tokenizer, prompt):
    tok = tokenizer([prompt]).to(DEVICE)
    feat = model.encode_text(tok)
    return feat / feat.norm(dim=-1, keepdim=True)


def compute_clip_i(model, preprocess, real_paths, gen_paths):
    """Trung binh cosine similarity giua MOI CAP (anh that, anh gen)."""
    real_embeds = [embed_image(model, preprocess, p) for p in real_paths]

    sims = []
    per_image_scores = []
    for g_path in gen_paths:
        g_emb = embed_image(model, preprocess, g_path)
        pair_sims = [(r_emb @ g_emb.T).item() for r_emb in real_embeds]
        avg_for_this_gen = sum(pair_sims) / len(pair_sims)
        per_image_scores.append({
            "gen_image": os.path.basename(g_path),
            "clip_i": round(avg_for_this_gen, 4),
        })
        sims.extend(pair_sims)

    overall = sum(sims) / len(sims)
    return overall, per_image_scores


def compute_clip_t(model, tokenizer, preprocess, items):
    """items: list of {"file":..., "prompt":...} tu metadata.json"""
    sims = []
    per_image_scores = []
    for item in items:
        img_path = os.path.join(GEN_IMG_DIR, item["file"])
        img_emb = embed_image(model, preprocess, img_path)
        txt_emb = embed_text(model, tokenizer, item["prompt"])
        sim = (img_emb @ txt_emb.T).item()
        sims.append(sim)
        per_image_scores.append({
            "gen_image": item["file"],
            "prompt": item["prompt"],
            "clip_t": round(sim, 4),
        })
    overall = sum(sims) / len(sims)
    return overall, per_image_scores


def load_metadata():
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    # Ho tro ca dinh dang cu (list) lan moi ({"lora_strength":..., "items":[...]})
    if isinstance(meta, list):
        return meta, None
    return meta["items"], meta.get("lora_strength")


def main():
    real_paths = sorted(
        p for ext in ("*.jpg", "*.jpeg", "*.png")
        for p in glob.glob(os.path.join(REAL_IMG_DIR, ext))
    )
    gen_paths = sorted(glob.glob(os.path.join(GEN_IMG_DIR, "*.png")))

    assert len(real_paths) > 0, f"Khong tim thay anh that trong {REAL_IMG_DIR}"
    assert len(gen_paths) > 0, (
        f"Khong tim thay anh generate trong {GEN_IMG_DIR}. "
        "Chay 04_generate_samples.py truoc."
    )
    assert os.path.exists(META_PATH), f"Thieu {META_PATH}. Chay lai 04_generate_samples.py."

    items, lora_strength = load_metadata()
    model, preprocess, tokenizer = load_model()

    print(f">>> So anh that (training set): {len(real_paths)}")
    print(f">>> So anh generate (test set): {len(gen_paths)}")

    print(">>> Tinh CLIP-I (subject fidelity)...")
    clip_i_score, clip_i_detail = compute_clip_i(model, preprocess, real_paths, gen_paths)

    print(">>> Tinh CLIP-T (prompt fidelity)...")
    clip_t_score, clip_t_detail = compute_clip_t(model, tokenizer, preprocess, items)

    result = {
        "clip_model": f"{MODEL_NAME} / openai",
        "lora_strength": lora_strength,
        "num_real_images": len(real_paths),
        "num_generated_images": len(gen_paths),
        "CLIP-I_overall": round(clip_i_score, 4),
        "CLIP-T_overall": round(clip_t_score, 4),
        "clip_i_per_image": clip_i_detail,
        "clip_t_per_image": clip_t_detail,
    }

    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    with open(RESULT_PATH, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("")
    print("==========================================")
    print(f"  CLIP-I (subject fidelity) : {clip_i_score:.4f}")
    print(f"  CLIP-T (prompt fidelity)  : {clip_t_score:.4f}")
    print(f"  Chi tiet luu tai: {RESULT_PATH}")
    print("==========================================")
    print("")
    print("  Tham khao muc diem (SD1.5 LoRA, CLIP ViT-B/32):")
    print("    CLIP-I ~0.6-0.8   -> subject fidelity tot")
    print("    CLIP-T ~0.25-0.32 -> bam sat prompt tot")
    print("    (CLIP-I qua cao ~>0.9 co the la dau hieu overfit)")


if __name__ == "__main__":
    main()
