# Assignment 2.6 — Flow Matching trên dữ liệu 2D 8-mode Gaussian

## Mục tiêu

Thí nghiệm cài đặt **Flow Matching (FM) từ đầu** bằng NumPy trên phân phối hỗn hợp 8 Gaussian trong không gian 2D. Mô hình học trường vận tốc `VelocityNet(x_t, t) -> v`, sinh mẫu bằng Euler với 1, 5, 10 và 50 bước, rồi so sánh độ thẳng quỹ đạo với baseline DDPM/DDIM. Phần bonus thực hiện **Reflow**.

> Các hình PNG trong thư mục này là kết quả đã sinh sẵn. README này chỉ tổng hợp và diễn giải kết quả, không chạy lại thí nghiệm.

## Cấu trúc mã nguồn

| File | Nội dung |
|---|---|
| `data.py` | Sinh dữ liệu 8 Gaussian, bán kính 3 và độ lệch chuẩn 0.15. |
| `nn.py` | MLP NumPy và Adam viết thủ công. |
| `flow_matching.py` | Huấn luyện FM và Euler sampler. |
| `ddpm.py` | DDPM noise-prediction baseline và deterministic DDIM sampler. |
| `metrics.py` | Hai metric độ thẳng: normalized deviation và path-length ratio. |
| `run_experiment.py` | Chạy toàn bộ thí nghiệm, vẽ hình và thực hiện Reflow. |

## Phương pháp

Với noise `x0 ~ N(0, I)` và dữ liệu `x1 ~ p_data`, dùng đường nội suy tuyến tính:

```text
x_t = (1 - t) x0 + t x1,     t ~ Uniform(0, 1)
u_t = x1 - x0
L_FM = E[||VelocityNet(x_t, t) - u_t||²]
```

Sau huấn luyện, giải ODE `dx/dt = v_theta(x,t)` từ `t=0` đến `t=1` bằng Euler. `STEP_COUNTS = [1, 5, 10, 50]` là các số bước được yêu cầu.

### Lưu ý kỹ thuật đã sửa

Trong phiên bản đầu, gradient đã bị chia cho batch size hai lần: một lần trong `dout = 2 * diff / batch_size` và một lần trong `MLP.backward`. `nn.py` hiện chỉ dùng tổng gradient trong `backward`, nên gradient đúng với loss trung bình theo batch. Việc này không đòi hỏi phải diễn giải lại các figure đã có; với Adam, việc scale đồng đều gradient thường ít ảnh hưởng đến quỹ đạo tối ưu, nhưng bản mã hiện tại đúng công thức hơn.

`run_experiment.py` cũng tạo thư mục output tự động. Mặc định kết quả được ghi vào `outputs/`; có thể đặt biến môi trường `OUT_DIR` để đổi thư mục đích.

## Kết quả và hình dùng trong báo cáo

| Hình | Nội dung | Kết luận nên nêu |
|---|---|---|
| `00_training_losses.png` | Loss FM và DDPM theo số iteration. | Cả hai loss giảm nhanh rồi ổn định. |
| `01_fm_trajectories.png` | Euler trajectories của FM với 1/5/10/50 bước. | FM đưa noise về tám mode dữ liệu; quỹ đạo cong nhẹ khi tăng số bước. |
| `02_ddpm_trajectories.png` | DDIM trajectories của DDPM, dùng cùng noise ban đầu. | Quỹ đạo DDPM/DDIM cong hơn rõ rệt. |
| `03_straightness_comparison.png` | So sánh định lượng FM và DDPM/DDIM. | Với 5, 10, 50 bước, FM có normalized deviation và path-length ratio thấp hơn DDPM/DDIM, do đó thẳng hơn. |
| `04_one_step_quality.png` | Chất lượng sinh mẫu one-step của FM và DDPM/DDIM. | Minh họa trade-off giữa rất ít bước lấy mẫu và chất lượng sample. |
| `05_reflow_trajectories.png` | Euler trajectories sau Reflow. | Trajectory gần đường thẳng hơn đáng kể. |
| `06_reflow_comparison.png` | Curvature và path-length ratio trước/sau Reflow. | Reflow làm normalized deviation gần 0 và path-length ratio gần 1. |
| `07_reflow_one_step_quality.png` | One-step samples sau Reflow. | Reflow cải thiện khả năng lấy mẫu bằng rất ít bước. |

Không sử dụng chỉ số curvature tại **1 step** để kết luận: quỹ đạo khi đó chỉ có điểm đầu và điểm cuối, nên normalized deviation bằng 0 theo định nghĩa, dù đường đi thực tế chưa được quan sát ở các điểm trung gian. So sánh có ý nghĩa là tại 5, 10 và 50 bước.

## Bonus: Reflow

1. Dùng FM đã huấn luyện giải Euler 100 bước để tạo **10,000 cặp xác định** `(noise, sample)`.
2. Huấn luyện một VelocityNet mới trên các cặp ghép này với cùng FM loss.
3. So sánh quỹ đạo trước/sau bằng hai metric trong `metrics.py`.

Hình `06_reflow_comparison.png` cho thấy sau Reflow, cả normalized deviation lẫn path-length ratio đều gần giá trị của đường thẳng lý tưởng (0 và 1 tương ứng). Đây là bằng chứng định lượng cho bonus.

## Cách chạy lại (chỉ khi cần)

```bash
python run_experiment.py
```

Mã cần NumPy và Matplotlib. Kết quả PNG sẽ nằm trong `outputs/` (hoặc thư mục do `OUT_DIR` chỉ định).
