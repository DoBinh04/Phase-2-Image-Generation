# Assignment 2.3 — Thay vòng lấy mẫu DDPM bằng DDIM

## 1. Mục tiêu

**Đề bài:** *Reuse the model trained in Assignment 2 and replace the sampling loop with DDIM.*

Bài làm tái sử dụng checkpoint U-Net đã huấn luyện ở Assignment 2.2 và **không huấn luyện lại mô hình**. Phần thay đổi duy nhất ở pha suy luận là thay vòng reverse diffusion DDPM tuần tự đủ `T = 1000` bước bằng DDIM, có thể lấy mẫu trên một dãy timestep thưa hơn (50 bước trong thí nghiệm).

## 2. Thiết lập thí nghiệm

| Thành phần | Cấu hình |
| --- | --- |
| Dataset / mô hình kế thừa | MNIST, U-Net từ Assignment 2.2 |
| Checkpoint | `checkpoints/ddpm_last.pt` |
| Điều kiện lớp | Có, nhãn chữ số `0–9` (classifier-free guidance) |
| Noise schedule | Cosine |
| Số timestep khi huấn luyện | `T = 1000` |
| `base_ch` | 64 |
| Batch size khi huấn luyện 2.2 | 128 |
| Số epoch huấn luyện 2.2 | 20 |
| DDIM inference | 50 bước, `eta = 0.0` |
| Guidance scale | 3.0 |
| Thiết bị chạy | CUDA (Google Colab) |

Checkpoint và toàn bộ siêu tham số kiến trúc/scheduler phải khớp với Assignment 2.2. DDIM không yêu cầu checkpoint mới vì nó chỉ thay đổi cách đi ngược từ nhiễu sang ảnh.

## 3. Cài đặt DDIM

### DDPM gốc

DDPM khởi tạo từ `x_T ~ N(0, I)` và chạy từng bước `t = T-1, ..., 0`. Ở mỗi bước, nó dự đoán nhiễu rồi cộng thêm nhiễu ngẫu nhiên theo posterior (trừ bước cuối). Vì vậy với mô hình này, DDPM cần 1000 lần gọi U-Net để sinh một batch ảnh.

### DDIM thay thế

`GaussianDiffusion.ddim_sample(...)` chọn một dãy giảm gồm `S` timestep từ `[T-1, ..., 0]`; ở đây dùng `S = 50`. Tại timestep `t`, từ dự đoán nhiễu `epsilon_theta(x_t, t)`, trước hết ước lượng ảnh sạch:

```text
x0_pred = (x_t - sqrt(1 - alpha_bar_t) * epsilon_theta) / sqrt(alpha_bar_t)
```

Sau đó cập nhật trực tiếp tới timestep trước đó `t_prev`:

```text
sigma = eta * sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t)
                   * (1 - alpha_bar_t / alpha_bar_prev))

x_prev = sqrt(alpha_bar_prev) * x0_pred
       + sqrt(1 - alpha_bar_prev - sigma^2) * epsilon_theta
       + sigma * z
```

Với `eta = 0`, `sigma = 0`, nên không có nhiễu ngẫu nhiên sau khi đã cố định `x_T`; DDIM trở thành deterministic. Điều này giúp lấy mẫu nhanh hơn đáng kể: 50 lần gọi U-Net thay vì 1000 lần, tức ít hơn **20 lần**.

Với mô hình conditional, nhiễu dự đoán dùng classifier-free guidance:

```text
epsilon = epsilon_uncond + w * (epsilon_cond - epsilon_uncond)
```

Trong đó `w = 3.0` là `guidance_scale`.

## 4. Cấu trúc thư mục

| File | Vai trò |
| --- | --- |
| `scheduler.py` | Tạo beta, alpha và `alpha_bar` cho linear/cosine schedule; giữ nguyên để khớp Assignment 2.2. |
| `model.py` | U-Net với ResNet blocks, sinusoidal time embedding và class conditioning; dùng để nạp checkpoint cũ. |
| `diffusion.py` | Giữ `q_sample`, loss và DDPM sampler làm baseline; bổ sung `ddim_step` và `ddim_sample`. |
| `sample_ddim.py` | Nạp checkpoint, sinh ảnh bằng DDIM; có tùy chọn chạy DDPM để so sánh. |
| `verify_ddim.py` | Kiểm thử công thức DDIM và các tính chất thực nghiệm. |
| `assingment23.ipynb` | Notebook Colab đã chạy, chứa log và ảnh kết quả. |

## 5. Kiểm chứng cài đặt

Khi chạy từ thư mục `assignment23`, đặt checkpoint của Assignment 2.2 tại
`checkpoints/ddpm_last.pt` (hoặc thay bằng đường dẫn tương đối phù hợp).

Lệnh đã chạy:

```bash
python verify_ddim.py \
  --ckpt checkpoints/ddpm_last.pt \
  --conditional \
  --schedule cosine \
  --timesteps 1000 \
  --base_ch 64
```

Kết quả trong notebook:

| Kiểm thử | Kết quả |
| --- | --- |
| Test 1 — `ddim_step` tái tạo quỹ đạo forward khi dùng exact noise, `eta=0` | PASS; max absolute error `9.42e-06` |
| Test 2 — phương sai DDIM `eta=1` bằng posterior variance DDPM tại các timestep kề nhau | PASS; max relative error `1.66e-04` |
| Test 3 — DDIM `eta=0` deterministic khi cùng `x_T` | PASS; max absolute difference `0.00e+00` |
| Test 4 — tăng số DDIM steps làm kết quả gần reference 200 bước hơn | PASS |

Test 3 bật deterministic CUDA để đo đúng tính xác định của DDIM, không bị ảnh hưởng bởi kernel convolution không xác định của GPU.

Sai số trung bình tuyệt đối so với output reference 200 bước trong Test 4:

| DDIM steps | 5 | 10 | 20 | 50 | 100 |
| ---: | ---: | ---: | ---: | ---: |
| Mean absolute error | 0.5126 | 0.4645 | 0.3774 | 0.2577 | 0.1570 |

Sai số giảm liên tục khi tăng số bước, đúng với kỳ vọng hội tụ của DDIM về quỹ đạo lấy mẫu dày hơn.

## 6. Sinh ảnh và so sánh

Sinh 8 ảnh cho mỗi chữ số, tổng 80 ảnh, bằng DDIM 50 bước:

```bash
python sample_ddim.py \
  --ckpt checkpoints/ddpm_last.pt \
  --conditional --schedule cosine --timesteps 1000 --base_ch 64 \
  --n_per_class 8 --guidance_scale 3.0 \
  --ddim_steps 50 --eta 0.0 --out_dir samples
```

Output: `samples/ddim_50steps_eta0.0.png`.

Chạy DDPM baseline từ cùng `x_T` và lưu ảnh so sánh:

```bash
python sample_ddim.py \
  --ckpt checkpoints/ddpm_last.pt \
  --conditional --schedule cosine \
  --n_per_class 8 --guidance_scale 3.0 \
  --ddim_steps 50 --eta 0.0 \
  --compare_ddpm --out_dir samples
```

Các file được tạo:

- `samples/ddim_50steps_eta0.0.png`: ảnh DDIM 50 bước.
- `samples/ddpm_1000steps.png`: ảnh baseline DDPM 1000 bước.
- `samples/ddpm_vs_ddim.png`: ảnh ghép so sánh. Mỗi phương pháp chiếm 10 hàng: 10 hàng đầu là DDPM và 10 hàng sau là DDIM; trong mỗi nhóm, các hàng lần lượt tương ứng nhãn 0–9 và mỗi hàng có 8 mẫu.

Các ảnh kết quả đã được nhúng trực tiếp trong notebook [assingment23.ipynb](assingment23.ipynb), vì vậy GitHub sẽ render chúng khi mở notebook. Nếu muốn hiện ảnh trực tiếp trong README, cần export các file trong `samples/` từ Colab, thêm chúng vào thư mục `assignment23/samples/` và commit cùng repository; khi đó có thể dùng đường dẫn tương đối như `![DDPM và DDIM](samples/ddpm_vs_ddim.png)`.

### Phân tích `ddpm_vs_ddim.png`

Hai nửa ảnh đều sinh chữ số theo đúng nhóm nhãn nhờ class conditioning và classifier-free guidance. Tuy nhiên, các ô cùng vị trí không cần giống hệt nhau dù hai sampler bắt đầu từ cùng `x_T`: DDPM thêm nhiễu mới ở từng bước reverse, nên tạo một quỹ đạo ngẫu nhiên 1000 bước; DDIM với `eta=0` không thêm nhiễu ở các bước reverse, do đó đi theo quỹ đạo xác định chỉ với 50 bước.

Vì DDIM bỏ qua nhiều timestep, một số chữ số có thể khác về nét bút, độ sắc hoặc chi tiết so với DDPM 1000 bước. Đây là đánh đổi giữa tốc độ và chất lượng xấp xỉ: DDIM giảm khoảng 20 lần số lần gọi U-Net nhưng vẫn giữ được cấu trúc chữ số. Các nét chưa thật mượt, vùng mờ hoặc mẫu chưa rõ ràng (nếu có) chủ yếu phản ánh giới hạn của checkpoint MNIST sau 20 epoch và việc dùng 50 bước, không phải mô hình được huấn luyện lại cho DDIM.

## 7. Kết luận

Đã tái sử dụng thành công U-Net checkpoint từ Assignment 2.2 để thay sampler DDPM bằng DDIM. Công thức DDIM được kiểm chứng bằng các kiểm thử đại số và thực nghiệm; cả 4 kiểm thử đều PASS. Với `eta=0`, kết quả có tính lặp lại khi giữ nguyên nhiễu ban đầu. DDIM sinh mẫu conditional MNIST trong 50 bước thay vì 1000 bước của DDPM, giảm số lần suy luận U-Net khoảng 20 lần mà không cần huấn luyện lại mô hình.
