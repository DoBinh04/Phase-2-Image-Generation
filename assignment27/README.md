# Assignment 2.7 — Inference FLUX.1, FLUX.2 và Qwen-Image bằng Diffusers

## 1. Mục tiêu

Triển khai suy luận text-to-image bằng thư viện Hugging Face Diffusers cho ba họ mô hình:

| Mô hình | Tệp triển khai | Môi trường chạy | Kết quả đã lưu |
|---|---|---|---|
| FLUX.1-dev | `flux1.ipynb` | Kaggle, 2× Tesla T4 (mỗi GPU 14.56 GB) | `results/flux1.png` (1024×1024) |
| FLUX.2-klein-4B | `flux2.ipynb` (cell 5) | Kaggle, 2× Tesla T4 | `results/flux2_klein.png` (512×512) |
| Qwen-Image | `qwen_image_vastai_inference.py` | Vast.ai, RTX 3090/4090 | `qwen_image_output.png` (1024×1024) |

> Ảnh FLUX được trích trực tiếp từ output đã lưu trong notebook và đặt trong `results/`, vì vậy README hiển thị được khi tải lên GitHub. Việc rà soát dựa trên mã nguồn và output notebook; không chạy lại tải model hoặc inference trên máy cục bộ.

## 2. Cấu trúc thư mục

```text
.
├── flux1.ipynb                     # FLUX.1-dev, 4-bit NF4
├── flux2.ipynb                     # FLUX.2-klein-4B và thử nghiệm FLUX.2-dev
├── qwen_image_vastai_inference.py  # Qwen-Image, chương trình dòng lệnh
├── qwen_image_output.png           # Kết quả Qwen-Image 1024×1024
├── results/
│   ├── flux1.png                    # Kết quả FLUX.1-dev 1024×1024
│   └── flux2_klein.png              # Kết quả FLUX.2-klein-4B 512×512
└── README.md
```

## 3. Cách triển khai và tối ưu bộ nhớ

### FLUX.1-dev

Notebook tải `black-forest-labs/FLUX.1-dev`. Transformer được lượng tử hóa 4-bit NF4 bằng `BitsAndBytesConfig`; text encoders giữ ở FP16. Pipeline dùng `enable_model_cpu_offload()`, đồng thời bật VAE slicing và tiling để giảm VRAM đỉnh.

Cấu hình inference đã chạy:

- Prompt: `a tiny astronaut hatching from an egg on the moon`
- Kích thước: 1024×1024
- Số bước: 28
- Guidance scale: 3.5
- Seed: 0
- Phiên bản ghi trong notebook: Diffusers 0.35.1, Transformers 4.55.4, Accelerate 1.10.1, BitsAndBytes 0.46.1, PyTorch 2.10.0+cu128.

Output của notebook xác nhận CUDA khả dụng trên 2 GPU Tesla T4 và kết thúc với `Saved: /kaggle/working/outputs/flux1.png`.

![Kết quả FLUX.1-dev](results/flux1.png)

### FLUX.2-klein-4B

Cell thực thi chính sử dụng `Flux2KleinPipeline` với model `black-forest-labs/FLUX.2-klein-4B`, `bfloat16` và CPU model offload trên GPU 0. Khi `IMAGE_PATH = None`, pipeline hoạt động ở chế độ text-to-image; có thể gán đường dẫn ảnh để chuyển sang image editing. VAE slicing/tiling được bật nếu pipeline hỗ trợ.

Cấu hình inference đã chạy:

- Prompt: `a cat holding a sign that says "hello world", cinematic lighting, highly detailed, photorealistic`
- Kích thước: 512×512
- Số bước: 4 (biến thể distilled)
- Guidance scale: 1.0
- Seed: 0
- Phiên bản ghi trong notebook: Diffusers 0.40.0, Transformers 5.17.0, Accelerate 1.15.0, Hugging Face Hub 1.32.0.

Notebook ghi nhận model load thành công và sinh ảnh 512×512. Cell 6 là phương án thử nghiệm riêng cho `diffusers/FLUX.2-dev-bnb-4bit` với group offload; cell này **chưa được chạy**, vì vậy không xem đó là kết quả đã xác nhận.

![Kết quả FLUX.2-klein-4B](results/flux2_klein.png)

### Qwen-Image

Script dùng `Qwen/Qwen-Image` với `PipelineQuantizationConfig`: transformer và text encoder được lượng tử hóa BitsAndBytes 4-bit NF4, tính toán bằng BF16. Sau khi load pipeline, model CPU offload được bật để giảm VRAM. Cách kết hợp này khớp với hướng dẫn Diffusers cho Qwen-Image; tài liệu nêu ví dụ 4-bit + offload dùng khoảng 12.54 GB VRAM.

Script còn vô hiệu hóa `_caching_allocator_warmup`, một tối ưu khởi tạo có thể tính theo kích thước weight trước lượng tử hóa và gây cấp phát VRAM dư. Việc tắt này chỉ ảnh hưởng tốc độ nạp model, không thay đổi ảnh đầu ra.

Mặc định:

- Prompt: `a tiny astronaut hatching from an egg on the moon, ultra HD, 4K, cinematic composition`
- Kích thước: 1024×1024
- Số bước: 25
- True CFG scale: 4.0
- Seed: 0
- Ảnh đầu ra: `qwen_image_output.png`

![Kết quả Qwen-Image](qwen_image_output.png)

## 4. Chạy Qwen-Image trên Vast.ai

### Yêu cầu

- Một GPU NVIDIA có CUDA; RTX 3090 hoặc RTX 4090 (24 GB VRAM) là cấu hình mục tiêu.
- Python, CUDA/PyTorch tương thích và quyền truy cập model Hugging Face (nếu model yêu cầu xác nhận điều khoản).
- RAM hệ thống đủ cho CPU offload.

Tạo `.env` cạnh script, không đưa tệp này lên Git:

```env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

Cài dependencies. Nên chạy trong môi trường sạch và lưu lại phiên bản đã cài để tái lập kết quả:

```bash
pip install -U diffusers transformers accelerate safetensors bitsandbytes python-dotenv huggingface_hub
```

Sinh ảnh với cấu hình mặc định:

```bash
python qwen_image_vastai_inference.py
```

Ví dụ tùy chỉnh:

```bash
python qwen_image_vastai_inference.py \
  --prompt "một thành phố ven biển lúc hoàng hôn, phong cách điện ảnh" \
  --steps 25 --cfg 4.0 --seed 42 \
  --width 1024 --height 1024 \
  --out ket_qua.png
```

`--no-offload` tăng tốc khi VRAM thực sự còn dư; `--no-quantize` không phù hợp cho một GPU 24 GB theo chú thích của script.

## 5. Rà soát mã nguồn

### Điểm đạt

- Cả ba triển khai đều gọi pipeline từ Diffusers, dùng `torch.inference_mode()` tại lúc sinh ảnh và lưu `images[0]` ra PNG.
- FLUX.1 và Qwen-Image áp dụng lượng tử hóa 4-bit; các notebook FLUX bật các cơ chế giảm bộ nhớ phù hợp với GPU T4/24 GB.
- Seed, prompt, độ phân giải, steps và guidance đều được khai báo tường minh, hỗ trợ tái lập inference.
- Script Qwen có kiểm tra CUDA trước khi tải model, CLI đầy đủ và không hard-code Hugging Face token.

### Điểm cần lưu ý

1. `qwen_image_vastai_inference.py` không khóa phiên bản dependency. API `PipelineQuantizationConfig` phụ thuộc Diffusers tương đối mới; nên lưu `requirements.txt`/`pip freeze` của môi trường chạy thực tế để tránh lỗi tương thích khi cài bản mới hơn.
2. Token phải chỉ nằm trong `.env`; nên thêm `.env` vào `.gitignore` nếu repository được quản lý bằng Git.
3. FLUX.2 notebook có hai hướng triển khai. Kết quả đã xác minh là **FLUX.2-klein-4B ở cell 5**; cell 6 cho FLUX.2-dev cần được chạy và lưu output/error riêng trước khi báo cáo là hoàn thành.
4. Các đường dẫn `/kaggle/working/...` chỉ hợp lệ trong Kaggle. Khi chuyển notebook sang máy khác cần đổi `OUTPUT_DIR`.

## 6. Kết luận

Assignment đã có ba triển khai inference bằng Diffusers. Hai notebook FLUX có log hoàn tất trên Kaggle 2× T4; Qwen-Image được đóng gói thành script CLI tối ưu 4-bit + CPU offload.
