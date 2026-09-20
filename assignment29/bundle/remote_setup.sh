#!/usr/bin/env bash
# Stage 0: chuan bi moi truong tren vast.ai (RTX 3090, ComfyUI co san)
set -euo pipefail

log(){ echo "[setup $(date +%H:%M:%S)] $*"; }

# --- 1. Tim ComfyUI ---
CUI="${COMFYUI_DIR:-}"
if [ -z "$CUI" ]; then
  for c in /opt/ComfyUI /workspace/ComfyUI /root/ComfyUI /ComfyUI /opt/workspace-internal/ComfyUI; do
    [ -f "$c/main.py" ] && CUI="$c" && break
  done
fi
[ -z "$CUI" ] && CUI=$(find / -maxdepth 5 -name main.py -path '*ComfyUI*' 2>/dev/null | head -1 | xargs -r dirname)
[ -z "$CUI" ] && { echo "KHONG TIM THAY ComfyUI"; exit 1; }
log "ComfyUI: $CUI"
echo "$CUI" > /root/.comfyui_path

cd "$CUI"
log "ComfyUI commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

# --- 2. Tai model FLUX.1 Kontext [dev] ---
mkdir -p models/diffusion_models models/text_encoders models/vae
HF="https://huggingface.co/Comfy-Org/flux1-kontext-dev_ComfyUI/resolve/main/split_files"

dl(){ # url dest
  local url="$1" dst="$2"
  if [ -s "$dst" ]; then log "co san: $(basename "$dst") ($(du -h "$dst"|cut -f1))"; return; fi
  log "tai $(basename "$dst") ..."
  curl -fL --retry 5 --retry-delay 3 -C - -o "$dst.part" "$url" && mv "$dst.part" "$dst"
  log "xong: $(basename "$dst") ($(du -h "$dst"|cut -f1))"
}

dl "$HF/diffusion_models/flux1-dev-kontext_fp8_scaled.safetensors" models/diffusion_models/flux1-dev-kontext_fp8_scaled.safetensors
dl "$HF/text_encoders/clip_l.safetensors"                          models/text_encoders/clip_l.safetensors
dl "$HF/text_encoders/t5xxl_fp8_e4m3fn_scaled.safetensors"         models/text_encoders/t5xxl_fp8_e4m3fn_scaled.safetensors
dl "$HF/vae/ae.safetensors"                                        models/vae/ae.safetensors

# --- 3. Deps cho post-processing (parsing / QA) ---
log "cai python deps ..."
pip install -q --no-input transformers pillow scikit-image scipy requests 2>&1 | tail -2 || true

# --- 4. Khoi dong ComfyUI (neu chua chay) ---
if curl -s -m 3 http://127.0.0.1:8188/system_stats >/dev/null 2>&1; then
  log "ComfyUI dang chay san tren :8188"
else
  log "khoi dong ComfyUI ..."
  nohup python main.py --listen 0.0.0.0 --port 8188 --preview-method none \
      > /root/comfyui.log 2>&1 &
  for i in $(seq 1 90); do
    curl -s -m 2 http://127.0.0.1:8188/system_stats >/dev/null 2>&1 && break
    sleep 2
  done
fi
curl -s http://127.0.0.1:8188/system_stats | head -c 600; echo
log "SETUP DONE"
