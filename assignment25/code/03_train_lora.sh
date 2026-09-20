#!/bin/bash
# ==========================================================
# 03_train_lora.sh
# Train LoRA cho subject "sks dog" (dataset dog6, 5-6 anh)
# Toi uu cho RTX 3090 24GB, SD1.5
# ==========================================================
set -e

WS=/workspace
source "$WS/sd-scripts/venv/bin/activate"
cd "$WS/sd-scripts"

mkdir -p "$WS/lora_train/output" "$WS/lora_train/logs"

# ---- Hyperparameter (sua o day de lam bang so sanh trong bao cao) ----
NETWORK_DIM=32          # rank LoRA. It anh -> 16-32 la du, cao hon de overfit
NETWORK_ALPHA=16        # thuong = dim/2
MAX_STEPS=1500          # 1200-1600 la vung hop ly cho 5-6 anh
LR=1e-4                 # unet lr
TE_LR=5e-5              # text encoder lr (thap hon unet)
BATCH=2
OUTPUT_NAME=sks_dog_lora

# LUU Y quan trong: clip_skip=1 (mac dinh cua SD1.5 goc).
# clip_skip=2 chi dung cho model anime; neu train clip_skip=2 ma infer
# bang diffusers (mac dinh =1) thi LoRA se lech, CLIP-I/CLIP-T deu tut.
accelerate launch --config_file "$WS/accelerate_config.yaml" train_network.py \
  --pretrained_model_name_or_path="$WS/models/v1-5-pruned-emaonly-fp16.safetensors" \
  --train_data_dir="$WS/lora_train/img" \
  --output_dir="$WS/lora_train/output" \
  --output_name="$OUTPUT_NAME" \
  --logging_dir="$WS/lora_train/logs" \
  --resolution=512,512 \
  --enable_bucket \
  --min_bucket_reso=256 \
  --max_bucket_reso=1024 \
  --network_module=networks.lora \
  --network_dim=$NETWORK_DIM \
  --network_alpha=$NETWORK_ALPHA \
  --train_batch_size=$BATCH \
  --learning_rate=$LR \
  --unet_lr=$LR \
  --text_encoder_lr=$TE_LR \
  --lr_scheduler=cosine_with_restarts \
  --lr_scheduler_num_cycles=3 \
  --lr_warmup_steps=100 \
  --max_train_steps=$MAX_STEPS \
  --mixed_precision=fp16 \
  --save_precision=fp16 \
  --save_every_n_epochs=20 \
  --save_model_as=safetensors \
  --sdpa \
  --cache_latents \
  --max_data_loader_n_workers=2 \
  --persistent_data_loader_workers \
  --seed=42 \
  --clip_skip=1 \
  --caption_extension=.txt \
  --shuffle_caption \
  --keep_tokens=1

echo ""
echo "=========================================="
echo "  TRAIN XONG."
echo "  LoRA output: $WS/lora_train/output/$OUTPUT_NAME.safetensors"
echo "  Buoc tiep theo: python3 04_generate_samples.py"
echo "=========================================="
