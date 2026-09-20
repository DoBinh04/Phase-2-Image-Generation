#!/bin/bash
# ==========================================================
# 02_prepare_data.sh
# Tai subject "dog6" tu Google DreamBooth dataset (CC-BY-4.0)
# va sap xep theo cau truc kohya_ss + viet caption
#
# Nguon: https://huggingface.co/datasets/google/dreambooth
# License: CC-BY-4.0
# Citation: Ruiz et al., "DreamBooth: Fine Tuning Text-to-Image
#           Diffusion Models for Subject-Driven Generation", CVPR 2023.
#
# Neu thu muc dich DA CO anh (ban tu upload) -> script bo qua buoc tai.
# Neu anh DA CO file .txt caption -> bo qua luon buoc auto-caption
# (khong ghi de caption ban da sua tay).
# ==========================================================
set -e

WS=/workspace
source "$WS/sd-scripts/venv/bin/activate"

# So '10' = so lan repeat moi anh trong 1 epoch (data it -> repeat nhieu)
# 'sks dog' = instance token (sks) + class name (dog)
TRAIN_DIR="$WS/lora_train/img/10_sks_dog"
mkdir -p "$TRAIN_DIR"

NUM_IMG=$(ls "$TRAIN_DIR"/*.jpg "$TRAIN_DIR"/*.jpeg "$TRAIN_DIR"/*.png 2>/dev/null | wc -l)

if [ "$NUM_IMG" -gt 0 ]; then
    echo ">>> [1/3] Da co $NUM_IMG anh san trong $TRAIN_DIR -> bo qua buoc tai dataset"
else
    echo ">>> [1/3] Tai subject 'dog6' tu Google DreamBooth dataset"
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='google/dreambooth',
    repo_type='dataset',
    allow_patterns='dataset/dog6/*',
    local_dir='$WS/dreambooth_data'
)
"
    echo ">>> [2/3] Sap xep vao cau truc thu muc kohya_ss"
    cp "$WS"/dreambooth_data/dataset/dog6/*.jpg "$TRAIN_DIR"/
fi

# Xoa rac Jupyter neu co (kohya se quet nham thu muc nay)
rm -rf "$TRAIN_DIR/.ipynb_checkpoints"

NUM_TXT=$(ls "$TRAIN_DIR"/*.txt 2>/dev/null | wc -l)
NUM_IMG=$(ls "$TRAIN_DIR"/*.jpg "$TRAIN_DIR"/*.jpeg "$TRAIN_DIR"/*.png 2>/dev/null | wc -l)

if [ "$NUM_TXT" -ge "$NUM_IMG" ] && [ "$NUM_TXT" -gt 0 ]; then
    echo ">>> [3/3] Da co du $NUM_TXT caption -> bo qua auto-caption BLIP"
else
    echo ">>> [3/3] Auto-caption bang BLIP (dua tren NOI DUNG THAT cua tung anh)"
    pip install --quiet transformers pillow

    TRAIN_DIR="$TRAIN_DIR" python3 << 'PYEOF'
import os
import glob
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

TRAIN_DIR = os.environ["TRAIN_DIR"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading BLIP captioning model...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(DEVICE)

img_paths = sorted(
    p for ext in ("*.jpg", "*.jpeg", "*.png")
    for p in glob.glob(os.path.join(TRAIN_DIR, ext))
)
for path in img_paths:
    txt_path = os.path.splitext(path)[0] + ".txt"
    if os.path.exists(txt_path):
        print(f"  {os.path.basename(path)} -> da co caption, bo qua")
        continue

    img = Image.open(path).convert("RGB")
    inputs = processor(img, return_tensors="pt").to(DEVICE)
    out = model.generate(**inputs, max_new_tokens=30)
    raw_caption = processor.decode(out[0], skip_special_tokens=True)

    # Chen instance token "sks dog": thay cum chung chung bang instance token,
    # dam bao model hoc dung: sks = subject nay, dog = class
    if "dog" in raw_caption:
        final_caption = raw_caption.replace("a dog", "a sks dog", 1)
        if "sks" not in final_caption:
            final_caption = final_caption.replace("dog", "sks dog", 1)
    else:
        final_caption = f"a sks dog, {raw_caption}"

    with open(txt_path, "w") as f:
        f.write(final_caption)

    print(f"  {os.path.basename(path)} -> {final_caption}")
PYEOF
fi

echo ""
echo "=========================================="
echo "  DATA CHUAN BI XONG: $TRAIN_DIR"
echo "  So anh: $(ls "$TRAIN_DIR"/*.jpg "$TRAIN_DIR"/*.jpeg "$TRAIN_DIR"/*.png 2>/dev/null | wc -l)"
echo ""
echo "  QUAN TRONG: mo tung file .txt va doi chieu voi anh .jpg tuong ung."
echo "  (BLIP thinh thoang mo ta sai mau/tu the/boi canh) - sua tay neu can."
echo ""
echo "  Buoc tiep theo: ./03_train_lora.sh"
echo "=========================================="
