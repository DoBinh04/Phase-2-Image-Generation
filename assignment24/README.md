# Assignment 2.4 — Manual CFG + DDIM với `diffusers`

## Mục tiêu

Thí nghiệm khảo sát ảnh hưởng của classifier-free guidance (CFG) lên ảnh sinh bởi Stable Diffusion. Bài làm **không dùng pipeline cấp cao để sinh ảnh**; thay vào đó, các thành phần của `diffusers` được nối thủ công và chạy bằng vòng lặp DDIM.

Prompt dùng cho toàn bộ bộ ảnh:

> `a majestic lion wearing a crown, studio lighting, detailed fur`

Thiết lập cố định: Stable Diffusion v1.5, DDIM 50 bước, seed `1234`, ảnh 512×512. Mỗi scale dùng cùng prompt và cùng latent noise ban đầu, vì thế guidance scale là biến duy nhất thay đổi.

## Các tệp

| Tệp | Nội dung |
|---|---|
| `assignment_2_4_cfg_ddim.py` | Chương trình suy luận CFG + DDIM thủ công và tạo ảnh/grid. |
| `assignment24.ipynb` | Notebook chạy thí nghiệm, đã điền prompt đúng với bộ ảnh. |
| `cfg_scale_comparison_grid.png` | So sánh các scale yêu cầu: 1, 3, 5, 7.5, 12, 20. |
| `cfg_scale_comparison_grid_2.png` | Thí nghiệm mở rộng scale 20–80 để xác nhận hiện tượng over-guidance. |
| `verify_manual_pipeline.py` | Kiểm tra tĩnh và kiểm tra chạy thực tế. |

## Quy trình suy luận thủ công

Chương trình thực hiện đúng bốn phần bắt buộc:

1. `CLIPTokenizer` và `CLIPTextModel` mã hoá prompt thành `cond_embeddings`.
2. Cùng encoder mã hoá chuỗi rỗng `""` thành `uncond_embeddings` (null prompt).
3. Với mỗi DDIM timestep, UNet chạy **hai forward pass riêng**:

   ```python
   noise_pred_uncond = unet(latent_model_input, t,
                             encoder_hidden_states=uncond_embeddings).sample
   noise_pred_cond = unet(latent_model_input, t,
                           encoder_hidden_states=cond_embeddings).sample
   noise_pred = noise_pred_uncond + guidance_scale * (
       noise_pred_cond - noise_pred_uncond
   )
   latents = scheduler.step(noise_pred, t, latents).prev_sample
   ```

4. Latent cuối cùng được chia cho `vae.config.scaling_factor` và giải mã bằng `vae.decode(...)` thành ảnh RGB.

`StableDiffusionPipeline` không được gọi trong đường sinh ảnh của chương trình chính.

## Cài đặt và chạy

```bash
pip install --upgrade torch diffusers transformers accelerate pillow numpy
```

Đăng nhập Hugging Face và bảo đảm có quyền truy cập model `stable-diffusion-v1-5/stable-diffusion-v1-5` trước khi chạy nếu model chưa có trong cache.

```bash
python assignment_2_4_cfg_ddim.py \
  --prompt "a majestic lion wearing a crown, studio lighting, detailed fur" \
  --model_id stable-diffusion-v1-5/stable-diffusion-v1-5 \
  --steps 50 \
  --seed 1234 \
  --scales 1 3 5 7.5 12 20 \
  --out_dir ./cfg_sweep_output
```

Kết quả gồm một ảnh `cfg_scale_{s}.png` cho mỗi scale và grid `cfg_scale_comparison_grid.png`.

## Kết quả và nhận xét

Đánh giá dưới đây dựa trên trực tiếp grid ảnh đã tạo với prompt, seed và model ở trên — không lấy mặc định 7.5 làm kết luận trước.

| Guidance scale | Quan sát |
|---:|---|
| 1 | Ảnh bị nhiễu/vỡ và không thể hiện rõ chủ thể. Prompt adherence rất yếu. |
| 3 | Sư tử đã rõ và tự nhiên hơn, nhưng vương miện chưa được thể hiện thuyết phục. |
| 5 | Lông và gương mặt ổn định; dấu hiệu vương miện vẫn nhẹ nên chưa bám đầy đủ prompt. |
| 7.5 | Ảnh sạch và cấu trúc mặt còn tự nhiên, nhưng gần đơn sắc; vương miện chưa đủ rõ nên bám prompt chưa tốt. |
| **12** | Cân bằng tốt nhất cho prompt này: vương miện vàng đã rõ, lông còn nhiều chi tiết và tổng thể vẫn khá tự nhiên. Tương phản tăng nhưng chưa phá vỡ chủ thể. |
| 20 | Màu sắc và vương miện rất nổi bật, nhưng ảnh đã thiên về minh hoạ stylized; khuôn mặt/vương miện có các chi tiết gắt và kém tự nhiên hơn. |

**Sweet spot của thí nghiệm này là `s = 12`.** So với `s = 7.5`, ảnh tại `s = 12` thể hiện tốt hơn chi tiết “wearing a crown” và vẫn giữ được lông, gương mặt cùng bố cục sư tử khá tự nhiên. Tại `s = 20`, tương phản và màu sắc bị đẩy mạnh khiến ảnh bắt đầu mất tính chân thực; đây là dấu hiệu over-guidance đầu tiên trong sweep chính. Thử nghiệm 20–80 củng cố nhận định: từ `s = 30` trở lên ảnh ngày càng bệt màu, tương phản cực đoan và hình dạng khuôn mặt bị cách điệu/broken.

## Kiểm tra mã nguồn

```bash
# Chỉ kiểm tra cấu trúc code, không tải model
python verify_manual_pipeline.py --static_only
```

Script kiểm tra sự vắng mặt của high-level pipeline trong chương trình chính, hai UNet call sites, công thức CFG rõ ràng và `scheduler.step(...)`. Chế độ đầy đủ còn đối chiếu output thủ công với `StableDiffusionPipeline` chỉ như một **reference test**, không phải đường sinh ảnh nộp bài:

```bash
python verify_manual_pipeline.py \
  --prompt "a majestic lion wearing a crown, studio lighting, detailed fur" \
  --model_id stable-diffusion-v1-5/stable-diffusion-v1-5 \
  --steps 50 --scale 7.5 --seed 1234
```
