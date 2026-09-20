#!/bin/bash
# ==========================================================
# 01_setup.sh - Cai dat moi truong train LoRA tren RTX 3090
# Chay 1 lan duy nhat sau khi SSH vao vast.ai instance
# Moi thu nam trong /workspace (KHONG dung ~ / /root)
# ==========================================================
set -e

WS=/workspace
SD=$WS/sd-scripts

echo ">>> [1/6] Cai goi he thong can thiet"
apt-get update -y
apt-get install -y git wget aria2

echo ">>> [2/6] Clone kohya sd-scripts (dung de train LoRA)"
if [ ! -d "$SD" ]; then
    git clone https://github.com/kohya-ss/sd-scripts.git "$SD"
fi
cd "$SD"

echo ">>> [3/6] Tao virtual environment rieng cho sd-scripts"
if [ ! -x "$SD/venv/bin/python" ]; then
    python3 -m venv "$SD/venv"
fi
source "$SD/venv/bin/activate"

echo ">>> [4/6] Cai pytorch cu121 + dependencies cua sd-scripts"
pip install --upgrade pip
# sd-scripts hien tai on dinh nhat voi torch 2.5.x
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
# Cac goi dung cho buoc 04 (sinh anh) va 05 (danh gia CLIP)
pip install open_clip_torch peft pillow
# LUU Y: KHONG cai xformers o day - ban xformers tren PyPI build cho torch moi hon,
# cai vao se hong import. Script train dung --sdpa (PyTorch SDPA) la du nhanh.

echo ">>> [5/6] Cau hinh accelerate (1 GPU, fp16)"
cat > "$WS/accelerate_config.yaml" << 'EOF'
compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
downcast_bf16: 'no'
gpu_ids: '0'
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF

echo ">>> [6/6] Chuan bi base model Stable Diffusion 1.5"
mkdir -p "$WS/models"
CKPT="$WS/models/v1-5-pruned-emaonly-fp16.safetensors"
if [ ! -e "$CKPT" ]; then
    # Image nay da co san checkpoint SD1.5 trong /opt/model_store -> dung lai, khoi tai
    if [ -f /opt/model_store/v1-5-pruned-emaonly-fp16.safetensors ]; then
        ln -sf /opt/model_store/v1-5-pruned-emaonly-fp16.safetensors "$CKPT"
    else
        # Repo 'runwayml/stable-diffusion-v1-5' DA BI XOA khoi HuggingFace.
        # Mirror chinh thuc con song:
        aria2c -x 16 -o v1-5-pruned-emaonly-fp16.safetensors -d "$WS/models" \
          "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors"
    fi
fi
ls -l "$CKPT"

echo ""
echo "=========================================="
echo "  SETUP HOAN TAT."
echo "  Buoc tiep theo: chay ./02_prepare_data.sh"
echo "=========================================="
