# Báo cáo Assignment 2.2 — Xây dựng Diffusion Model từ đầu

## 1. Mục tiêu

Bài tập hiện thực mô hình khuyếch tán ảnh (DDPM) bằng **PyTorch thuần**, không sử dụng thư viện `diffusers`. Mô hình được huấn luyện trên tập **MNIST** và có phần mở rộng sinh ảnh theo nhãn chữ số (0–9) bằng *classifier-free guidance*.

| Yêu cầu | Hiện thực |
| --- | --- |
| Noise scheduler tuyến tính và cosine | `scheduler.py`: tính `beta`, `alpha = 1 - beta`, `alpha_bar` và các hệ số cần thiết. |
| Forward process | `GaussianDiffusion.q_sample(x0, t, noise)` trong `diffusion.py`. |
| U-Net tối giản | `model.py`: ResNet blocks, sinusoidal time embedding, skip connections và self-attention. |
| Vòng lặp huấn luyện | `train.py`: chọn bước thời gian, thêm nhiễu, dự đoán nhiễu và tối ưu MSE. |
| Sinh mẫu | `diffusion.py`: DDPM đủ `T` bước và DDIM nhanh. |
| Bonus: class conditioning | U-Net nhận nhãn 0–9; label dropout và classifier-free guidance. |

## 2. Dữ liệu và tiền xử lý

- Dataset chính: **MNIST**.
- Ảnh gốc `1 × 28 × 28` được đệm thành `1 × 32 × 32` để phù hợp với hai tầng downsample/up-sample của U-Net.
- Pixel được biến đổi từ `[0, 1]` sang `[-1, 1]`; ảnh sinh ra được đổi ngược về `[0, 1]` trước khi lưu.
- `data.py` cũng có DataLoader cho CIFAR-10. Khi dùng CIFAR-10 cần đổi số kênh đầu vào từ 1 sang 3 trong `train.py` và `sample.py`.

## 3. Phương pháp thực hiện

### 3.1. Noise Scheduler

Với mỗi bước thời gian `t`, scheduler tiền tính:

```text
beta_t                 : mức nhiễu tại bước t
alpha_t = 1 - beta_t
alpha_bar_t = ∏(alpha_i), i = 1..t
```

Hai lịch trình nhiễu được hỗ trợ:

- `linear`: `beta_t` tăng tuyến tính từ `1e-4` đến `0.02`.
- `cosine`: lịch trình cosine theo Nichol & Dhariwal, có giới hạn beta để ổn định tính toán.

Các tensor như `sqrt(alpha_bar_t)`, `sqrt(1 - alpha_bar_t)`, phương sai posterior và `sqrt(1 / alpha_t)` cũng được lưu sẵn để dùng trong huấn luyện và lấy mẫu.

### 3.2. Quá trình thuận

Từ ảnh sạch `x_0`, ảnh nhiễu ở bước `t` được lấy trực tiếp theo công thức:

```text
x_t = sqrt(alpha_bar_t) · x_0 + sqrt(1 - alpha_bar_t) · ε,
ε ~ N(0, I)
```

Hàm `q_sample` trả về cả `x_t` và nhiễu thật `ε`. Vì sử dụng dạng đóng, không cần thêm nhiễu tuần tự từ bước 1 đến `t`.

### 3.3. Kiến trúc U-Net

U-Net có ba mức phân giải:

```text
32 × 32 → 16 × 16 → 8 × 8 → 16 × 16 → 32 × 32
```

- Mỗi `ResBlock` dùng `GroupNorm → SiLU → Conv` hai lần, kèm residual connection.
- Bước thời gian `t` được mã hóa bằng **sinusoidal positional embedding**, sau đó qua MLP tạo time embedding.
- Time embedding được cộng vào từng ResBlock theo từng kênh, giúp mạng nhận biết mức nhiễu hiện tại.
- Một lớp self-attention được đặt tại bottleneck `8 × 8`.
- Khi bật điều kiện lớp, embedding nhãn được cộng với time embedding. Slot nhãn thứ 11 là *null token* cho trường hợp không điều kiện.

### 3.4. Huấn luyện

Ở mỗi batch, chương trình thực hiện:

1. Chọn `t` ngẫu nhiên đều trong `[0, T-1]`.
2. Sinh `x_t` từ `x_0` bằng `q_sample`.
3. U-Net nhận `(x_t, t, y)` và dự đoán nhiễu `ε_theta`.
4. Tối ưu hàm mất mát:

```text
L = MSE(ε_theta(x_t, t, y), ε)
```

Với mô hình có điều kiện, nhãn được thay bằng null token với xác suất `p_uncond = 0.1`. Nhờ đó, cùng một mạng học cả dự đoán có điều kiện và vô điều kiện để áp dụng classifier-free guidance.

### 3.5. Sinh ảnh

Quá trình sinh bắt đầu bằng Gaussian noise `x_T ~ N(0, I)` và khử nhiễu dần về `x_0`.

- `sample`: DDPM ancestral sampling, chạy đủ `T = 1000` bước.
- `ddim_sample`: DDIM, mặc định 50 bước để sinh nhanh hơn.
- Với sinh có điều kiện, nhiễu dự đoán được kết hợp theo:

```text
ε = ε_uncond + s · (ε_cond - ε_uncond)
```

Trong đó `s` là `guidance_scale`. Thử nghiệm dùng `s = 3.0`; mỗi hàng của ảnh kết quả tương ứng một chữ số từ 0 đến 9.

## 4. Cấu trúc mã nguồn

| Tệp | Nội dung |
| --- | --- |
| `scheduler.py` | Linear/cosine scheduler và các hệ số DDPM. |
| `diffusion.py` | Forward process, MSE loss, DDPM/DDIM sampling và classifier-free guidance. |
| `model.py` | Minimal U-Net, ResBlock, time embedding và class embedding. |
| `data.py` | DataLoader MNIST/CIFAR-10 và tiền xử lý. |
| `train.py` | Huấn luyện, lưu checkpoint và ảnh theo từng epoch. |
| `sample.py` | Nạp checkpoint và sinh lưới ảnh. |
| `smoke_test.py` | Kiểm tra nhanh scheduler, forward/backward và sampler. |
| `DDPM.ipynb` | Notebook đã chạy trên Google Colab, chứa lệnh chạy và ảnh kết quả. |

## 5. Thiết lập và cách chạy

Yêu cầu: Python, PyTorch và torchvision.

```bash
pip install torch torchvision
```

Kiểm tra nhanh toàn bộ thành phần (không tải dataset):

```bash
python smoke_test.py
```

Huấn luyện mô hình có điều kiện như cấu hình đã dùng trong notebook:

```bash
python train.py --epochs 20 --batch_size 128 --schedule cosine --conditional --p_uncond 0.1 --sample_every 5
```

Sinh lưới ảnh theo từng chữ số từ checkpoint:

```bash
python sample.py --ckpt checkpoints/ddpm_last.pt --conditional --schedule cosine --n_per_class 8 --guidance_scale 3.0 --sampler ddim --ddim_steps 50 --out samples/class_conditional_grid.png
```

Huấn luyện hoặc sinh ảnh không điều kiện chỉ cần bỏ cờ `--conditional`:

```bash
python train.py --epochs 20 --schedule cosine
python sample.py --ckpt checkpoints/ddpm_last.pt --n 64 --out samples/grid.png
```

## 6. Kết quả chạy thực nghiệm

Notebook `DDPM.ipynb` đã được chạy trên **Google Colab với Tesla T4** theo cấu hình: MNIST, U-Net 6.12 triệu tham số, cosine schedule, `T = 1000`, batch size 128, 20 epoch, class conditioning, `p_uncond = 0.1`.

Loss trung bình theo một số epoch:

| Epoch | 0 | 1 | 4 | 9 | 14 | 19 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MSE trung bình | 0.0883 | 0.0396 | 0.0321 | 0.0298 | 0.0290 | 0.0285 |

Loss giảm nhanh trong các epoch đầu và ổn định quanh `0.028–0.029` ở cuối quá trình huấn luyện. Notebook hiển thị các lưới mẫu tại epoch 4, 9, 14 và 19, cùng lưới sinh cuối cùng gồm **80 ảnh**: 10 hàng cho nhãn 0–9, mỗi hàng 8 ảnh độc lập.

`smoke_test.py` cũng đã chạy thành công cho cả hai scheduler (`linear`, `cosine`), hai chế độ (có/không điều kiện), và hai sampler (DDPM/DDIM). Mọi đầu ra mẫu đều có kích thước `(4, 1, 32, 32)` và backward pass hợp lệ.

## 7. Tham khảo

- Ho et al., *Denoising Diffusion Probabilistic Models* (2020).
- Nichol & Dhariwal, *Improved Denoising Diffusion Probabilistic Models* (2021).
- Song et al., *Denoising Diffusion Implicit Models* (2020).
- [Classifier-Free DDIM for MNIST](https://github.com/tatakai1/classifier_free_ddim/blob/main/Classifier_Free_DDIM_Mnist.ipynb).
