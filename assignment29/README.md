# Virtual Try-On bằng Flux Kontext — Thiết kế Workflow

> **Đề bài:** *Design a workflow for the virtual try-on task (input: person + clothing → output: person wearing the outfit) using Flux Kontext.*

Báo cáo này mô tả thiết kế, lý do lựa chọn kỹ thuật, và cách vận hành một workflow ComfyUI hoàn chỉnh cho bài toán thử đồ ảo dựa trên **FLUX.1 Kontext [dev]**.

> ⚠️ Workflow đã được kiểm tra tính hợp lệ về cấu trúc (topology, kiểu dữ liệu, link) nhưng **chưa chạy inference thực tế** — phần benchmark tốc độ/VRAM trong báo cáo là con số tham chiếu, cần đo lại trên máy đích.

---

## 1. Tóm tắt giải pháp

| | |
|---|---|
| **Input** | `person.png` (người mẫu, thấy rõ trang phục) + `garment.png` (ảnh đồ phẳng, nền trắng) |
| **Output** | `vton/01_tryon_xxxxx.png` — người trong ảnh A mặc bộ đồ trong ảnh B |
| **Model** | FLUX.1 Kontext [dev] fp8_scaled (12B, guidance-distilled) |
| **Kỹ thuật lõi** | Ghép 2 ảnh thành **canvas đôi** làm reference latent, rồi ra lệnh chỉnh sửa bằng ngôn ngữ tự nhiên |
| **Phần cứng** | 1× RTX 3090 24 GB (fp8 vừa VRAM, không cần offload) |
| **Custom node** | **Không cần** — toàn bộ 22 node đều là node lõi của ComfyUI |
| **File** | `flux_kontext_vton.json` (kéo-thả vào UI) · `flux_kontext_vton_api.json` (POST `/prompt`) |

---

## 2. Phân tích bài toán & lý do chọn cách tiếp cận

### 2.1 Flux Kontext không phải model VTON chuyên dụng

Đây là điểm cần nói rõ trước khi thiết kế. Các model VTON chuyên dụng (IDM-VTON, CatVTON, OOTDiffusion, Leffa) được huấn luyện trên dữ liệu **ghép cặp** `(người, món đồ, ảnh người mặc món đồ đó)`, nên chúng học được phép "warp" texture từ ảnh đồ phẳng sang bề mặt cơ thể.

Flux Kontext là model **in-context image editing** tổng quát. Thế mạnh và điểm yếu của nó ở bài này:

| Ưu | Nhược |
|---|---|
| Giữ identity/bố cục rất tốt so với các model edit khác | Không copy pixel-exact — **logo, chữ in, hoạ tiết nhỏ hay bị "diễn giải lại"** |
| Nhận lệnh ngôn ngữ tự nhiên → linh hoạt, không cần mask thủ công | Không có ràng buộc hình học → đôi khi áo bị "dán phẳng", không theo dáng người |
| Không cần human parsing / densepose ở bước sinh ảnh | Variance theo seed cao → cần chạy nhiều seed |
| Chạy được trên 1 GPU 24 GB, không cần pipeline nhiều model | Vẫn drift nhẹ ở mặt/nền dù prompt đã yêu cầu giữ nguyên |

**Kết luận thiết kế:** Kontext lo phần *"mặc đồ trông hợp lý"*; phần *"giữ đúng người"* phải do **mask compositing** đảm nhiệm (mục 7), và phần *"không ship ảnh hỏng"* do **vòng QA** đảm nhiệm (mục 8). Bỏ hai phần sau thì ra demo đẹp nhưng sản phẩm không dùng được.

### 2.2 Vì sao dùng "canvas đôi" thay vì multi-reference

Kontext [dev] nhận **một** ảnh điều kiện. Có hai cách đưa 2 ảnh vào:

| Cách | Mô tả | Đánh giá |
|---|---|---|
| **A. Stitch canvas** *(chọn)* | Ghép `[người ∣ đồ]` thành 1 ảnh rộng gấp đôi, prompt tham chiếu theo vị trí trái/phải | Model "nhìn" được cả hai trong cùng attention field → **bám texture đồ tốt hơn rõ rệt**. Phải cắt nửa trái ở output. |
| B. Chain `ReferenceLatent` | Nối 2 node `ReferenceLatent` liên tiếp, mỗi node một ảnh | Output đúng kích thước người, không cần crop. Nhưng độ trung thực hoạ tiết kém hơn. |

Workflow này chọn **A**, và bù lại nhược điểm bằng node `ImageCrop` ở cuối.

### 2.3 Vì sao ép kích thước cố định 608×832

`FluxKontextImageScale` tự snap ảnh về danh sách *preferred resolution* của Flux. Nếu để nó tự quyết, kích thước canvas đầu ra không đoán trước được → **không thể đặt giá trị crop cố định**.

Giải pháp: ép mỗi ảnh về **608×832** bằng `ImageScale` (crop `center`, không kéo méo), ghép lại ra đúng **1216×832** — vốn đã là một preferred resolution của Flux. Khi đó `FluxKontextImageScale` thành no-op an toàn, và `ImageCrop(608, 832, 0, 0)` luôn cắt chính xác nửa trái.

> Ảnh VITON-HD chuẩn là 768×1024 (tỉ lệ 0.75); 608×832 có tỉ lệ 0.731 → center-crop mất ~2.5% chiều ngang. Chấp nhận được. Muốn giữ nguyên khung thì đổi sang 624×832 (0.75) và sửa `ImageCrop.width` tương ứng — nhưng 1248×832 cũng là preferred res nên vẫn an toàn.

---

## 3. Kiến trúc workflow

```
┌─ 1. LOADERS ─────────────────────┐
│ UNETLoader   (Kontext fp8)       │──────────────────────────────┐
│ DualCLIPLoader (CLIP-L + T5-XXL) │──┐                           │
│ VAELoader    (ae.safetensors)    │──┼──────────┐                │
└──────────────────────────────────┘  │          │                │
                                      │          │                │
┌─ 2. INPUT ───────────────────────┐  │          │                │
│ LoadImage A (person)             │  │          │                │
│   └ ImageScale → 608×832  ───┐   │  │          │                │
│ LoadImage B (garment)         │  │  │          │                │
│   └ ImageScale → 608×832  ─┐  │   │ │          │                │
└────────────────────────────┼──┼──┘  │          │                │
                             │  │     │          │                │
┌─ 3. REFERENCE CANVAS ──────┼──┼──┐  │          │                │
│ ImageStitch [A│B] = 1216×832  ◄─┘  │          │                │
│   └ FluxKontextImageScale        │  │          │                │
│       └ VAEEncode  ◄─────────────┼──┼──────────┘                │
└──────────────┬───────────────────┘  │                           │
               │ LATENT               │ CLIP                      │
               │         ┌────────────┘                           │
┌─ 4. CONDITIONING ───────┼─────────────────────────────────┐     │
│ CLIPTextEncode (instruction + preservation clause)        │     │
│   ├─► ReferenceLatent(cond, latent) ─► FluxGuidance 2.5 ──┼──┐  │
│   └─► ConditioningZeroOut  (negative) ────────────────────┼─┐│  │
└───────────────────────────────────────────────────────────┘ ││  │
                                                              ││  │
┌─ 5. SAMPLING ────────────────────────────────────────────┐  ││  │
│ KSampler  model ◄────────────────────────────────────────┼──┼┼──┘
│           positive ◄─────────────────────────────────────┼──┘│
│           negative ◄─────────────────────────────────────┼───┘
│           latent_image ◄── (cùng latent của canvas)      │
│   └ VAEDecode                                            │
└────────┬─────────────────────────────────────────────────┘
         ├─► SaveImage  "vton/00_canvas"   ← debug: xem cả 2 nửa
         │
┌─ 6. POST ────────────────────────────────────────────────┐
│ ImageCrop(608, 832, 0, 0)   ← lấy nửa trái = người mặc đồ│
│   └ ImageUpscaleWithModel (4x-UltraSharp)                │
│       └ ImageScale → 912×1248                            │
│           └ SaveImage "vton/01_tryon"                    │
└──────────────────────────────────────────────────────────┘
```

### Bảng node

| # | Node | Vai trò | Tham số |
|---|---|---|---|
| 1 | `UNETLoader` | Flux Kontext dev | `flux1-dev-kontext_fp8_scaled.safetensors`, `fp8_e4m3fn` |
| 2 | `DualCLIPLoader` | Text encoder kép | `clip_l` + `t5xxl_fp8_e4m3fn_scaled`, type `flux` |
| 3 | `VAELoader` | VAE của Flux | `ae.safetensors` |
| 4 | `LoadImage` | **Input A** — người | `person.png` |
| 5 | `LoadImage` | **Input B** — trang phục | `garment.png` |
| 6 | `ImageScale` | Chuẩn hoá A | lanczos, 608×832, crop `center` |
| 7 | `ImageScale` | Chuẩn hoá B | lanczos, 608×832, crop `center` |
| 8 | `ImageStitch` | Ghép canvas đôi | `right`, match_size ✓, spacing 0, nền `white` |
| 9 | `FluxKontextImageScale` | Chốt preferred resolution | — |
| 10 | `VAEEncode` | Canvas → latent tham chiếu | — |
| 11 | `CLIPTextEncode` | Câu lệnh chỉnh sửa | xem mục 5 |
| 12 | `ReferenceLatent` | **Bơm canvas vào conditioning** — cơ chế in-context của Kontext | — |
| 13 | `FluxGuidance` | Độ bám prompt | `2.5` |
| 14 | `ConditioningZeroOut` | Negative rỗng (Flux distilled không dùng CFG thật) | — |
| 15 | `KSampler` | Sinh ảnh | seed 42, 20 steps, cfg 1.0, euler/simple, denoise 1.0 |
| 16 | `VAEDecode` | Latent → ảnh | — |
| 17 | `SaveImage` | Lưu canvas đầy đủ để soi lỗi | `vton/00_canvas` |
| 18 | `ImageCrop` | Cắt nửa trái | 608×832 @ (0,0) |
| 19–21 | `UpscaleModelLoader` → `ImageUpscaleWithModel` → `ImageScale` | Nâng nét 4× rồi hạ về 1.5× | `4x-UltraSharp.pth`, đích 912×1248 |
| 22 | `SaveImage` | **Output** | `vton/01_tryon` |

---

## 4. Ba chi tiết quyết định chất lượng

**a) `latent_image` của KSampler dùng chính latent của canvas, `denoise = 1.0`.**
Nhìn qua thì mâu thuẫn — denoise 1.0 nghĩa là sinh lại từ nhiễu hoàn toàn. Nhưng Kontext không dựa vào latent đầu vào để giữ ảnh; nó giữ ảnh qua **`ReferenceLatent`** (token ảnh tham chiếu được nối vào chuỗi conditioning). Đây là điểm khác bản chất với img2img. Đặt `denoise < 1.0` ở đây là sai cách dùng và làm ảnh bị mờ nhoè.

**b) `cfg = 1.0`, negative để rỗng.**
Flux là model guidance-distilled: guidance được truyền qua embedding (`FluxGuidance`), không qua classifier-free guidance. Tăng `cfg` lên >1 sẽ làm hỏng ảnh và tăng gấp đôi thời gian sinh. `ConditioningZeroOut` chỉ để cấp một conditioning hợp lệ cho slot `negative`.

**c) `spacing_width = 0`, nền `white`.**
Đường phân cách dày làm model hiểu canvas là "ảnh ghép đôi" và hay vẽ tiếp thành 2 khung riêng. Ghép sát mép cho kết quả ổn định hơn.

---

## 5. Prompt engineering

Prompt trong node 11:

```
Replace the top worn by the person on the left with the garment shown on the
right. Keep the person's face, hairstyle, skin tone, body shape, pose, hands,
the trousers, the shoes and the background completely unchanged. Match the
garment's fabric texture, print placement, logo and color exactly, with natural
folds and drape following the body pose, and lighting and shadows consistent
with the original scene.
```

Cấu trúc 3 phần, thứ tự quan trọng:

1. **Lệnh** — động từ `Replace` / `Change`, gọi đối tượng **theo vị trí** (`the person on the left`, `the garment on the right`). Tuyệt đối không dùng `it`, `this`, cũng không dùng `transform` hay `make it look like` — Kontext phản ứng với các động từ đó bằng cách vẽ lại toàn cảnh.
2. **Mệnh đề preservation** — liệt kê **cụ thể** thứ phải giữ. Đây là phần hay bị bỏ nhất và cũng là nguyên nhân số một khiến mặt người "đẹp lên" thành người khác. Có nghi ngờ gì thì cứ liệt kê thêm.
3. **Ràng buộc chất lượng** — texture, vị trí hoạ tiết, nếp gấp theo dáng, ánh sáng khớp cảnh.

**Biến thể theo loại đồ** — sửa cụm đầu và cụm preservation:

| Loại | Cụm lệnh | Thêm vào preservation |
|---|---|---|
| Áo | `Replace the top worn by...` | `the trousers, the shoes` |
| Quần/váy ngắn | `Replace the trousers worn by...` | `the top, the shoes` |
| Đầm liền | `Replace the outfit worn by... with the dress shown on the right` | `the shoes` |
| Áo khoác | `Add the jacket shown on the right onto the person on the left, worn open over their current top` | `the top underneath, the trousers` |

Nên mô tả món đồ bằng thuộc tính cụ thể thay vì chỉ "the garment": `the off-white short-sleeve cotton crew-neck t-shirt with black serif text "ATELIER" across the chest`. Trong pipeline thật, cụm này nên do một VLM sinh tự động từ ảnh đồ (caption có cấu trúc: category / sleeve / neckline / fit / color / material / graphic).

---

## 6. Tham số & hiệu năng

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| `FluxGuidance` | **2.5** | Khoảng làm việc 2.0–3.0. >3.5 → da nhựa, màu bệt |
| `steps` | **20** | 28 nếu cần nét hơn; >30 gần như không cải thiện |
| `cfg` | **1.0** | **Không đổi** |
| `sampler` / `scheduler` | `euler` / `simple` | `beta` cũng tốt, hơi tương phản hơn |
| `denoise` | **1.0** | Xem mục 4a |
| `weight_dtype` | `fp8_e4m3fn` | Bản bf16 (23 GB) sẽ tràn VRAM trên 3090 và chậm 3–4× |
| `seed` | `randomize` | **Chạy 3–4 seed rồi chọn** — variance cao |

Ước tính trên RTX 3090 24 GB @ canvas 1216×832: VRAM đỉnh **~19–21 GB**, **~35–55 s/ảnh**, lần chạy đầu cộng thêm ~60 s nạp model. *Cần đo lại thực tế.*

Model cần có (tổng ~17.5 GB):

```
ComfyUI/models/diffusion_models/flux1-dev-kontext_fp8_scaled.safetensors   11.9 GB
ComfyUI/models/text_encoders/clip_l.safetensors                           246 MB
ComfyUI/models/text_encoders/t5xxl_fp8_e4m3fn_scaled.safetensors           5.2 GB
ComfyUI/models/vae/ae.safetensors                                         335 MB
ComfyUI/models/upscale_models/4x-UltraSharp.pth                            67 MB
```

Nếu file text-encoder trên máy bạn tên là `t5xxl_fp8_e4m3fn.safetensors` (không có `_scaled`) thì sửa lại widget `clip_name2` của node 2 cho khớp.

---

## 7. Hậu xử lý bắt buộc: mask compositing

Output từ node 22 **chưa dùng được cho sản phẩm**. Dù prompt đã yêu cầu giữ nguyên, Kontext vẫn sinh lại **toàn bộ** khung hình, nên mặt sẽ "đẹp lên" đôi chút và nền lệch vài pixel. Người dùng nhận ra ngay đó không còn là ảnh của họ.

Cách xử lý: **chỉ nhận pixel sinh ra trong vùng quần áo**, phần còn lại lấy nguyên từ ảnh gốc.

```python
# pip install transformers pillow numpy scipy
import numpy as np, torch
from PIL import Image, ImageFilter
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from scipy.ndimage import binary_dilation

MODEL = "mattmdjaga/segformer_b2_clothes"
proc = SegformerImageProcessor.from_pretrained(MODEL)
net  = AutoModelForSemanticSegmentation.from_pretrained(MODEL).eval()

# nhãn: 4 Upper-clothes, 5 Skirt, 6 Pants, 7 Dress, 11 Face, 2 Hair
GARMENT, KEEP = {4, 5, 6, 7}, {2, 11}

def parse(img):
    with torch.no_grad():
        lo = net(**proc(images=img, return_tensors="pt")).logits
    up = torch.nn.functional.interpolate(lo, size=img.size[::-1],
                                         mode="bilinear", align_corners=False)
    return up.argmax(1)[0].numpy()

person = Image.open("person.png").convert("RGB").resize((608, 832), Image.LANCZOS)
gen    = Image.open("output/vton/01_tryon_00001_.png").convert("RGB").resize(person.size)

sp, sg = parse(person), parse(gen)
m  = np.isin(sp, list(GARMENT)) | np.isin(sg, list(GARMENT))   # hợp đồ cũ + đồ mới
m  = binary_dilation(m, iterations=4)                          # nuốt viền
m &= ~np.isin(sp, list(KEEP))                                  # khoá cứng mặt + tóc

soft = np.asarray(Image.fromarray((m * 255).astype("uint8"))
                  .filter(ImageFilter.GaussianBlur(4)), float)[..., None] / 255.0
out  = np.asarray(gen, float) * soft + np.asarray(person, float) * (1 - soft)
Image.fromarray(out.astype("uint8")).save("tryon_final.png")
```

**Trường hợp đổi silhouette** (đầm dài → áo ngắn, tay dài → tay cộc): vùng `đồ_cũ \ đồ_mới` bị hở ra và cần fill riêng —
- hở nền → inpaint bằng LaMa hoặc một pass Flux Fill;
- hở da (cánh tay, chân) → pass Kontext thứ hai giới hạn trong vùng đó: *"fill the exposed area with the person's bare arm, skin tone matching the visible skin, anatomically correct"*.

Bước này cố tình để **ngoài** graph ComfyUI vì nó cần custom node segmentation; tách ra giữ cho workflow chạy được trên ComfyUI sạch, không phụ thuộc.

---

## 8. Đánh giá chất lượng

### Metric tự động (dùng cho vòng auto-reject)

| Kiểm tra | Cách đo | Ngưỡng đề xuất |
|---|---|---|
| Giữ identity | ArcFace cosine(gen, gốc) | ≥ 0.85 |
| Đúng món đồ | DINOv2 / CLIP-I giữa ảnh đồ và crop vùng đồ trong ảnh ra | ≥ 0.80 |
| Giữ tư thế | MSE keypoint (DWPose, đã chuẩn hoá) | ≤ 0.02 |
| Giữ nền | SSIM tính ngoài mask chỉnh sửa | ≥ 0.97 |
| Lỗi giải phẫu / chữ sai | VLM chấm theo rubric 5 câu hỏi Y/N | 0 fail |

Sinh **K = 3–4 seed**, tính điểm tổng hợp có trọng số, chọn bản tốt nhất. Nếu cả K đều trượt: đổi seed và hạ `guidance` 0.5, tối đa 2 vòng, sau đó trả trạng thái `needs_review` thay vì trả ảnh hỏng — với VTON, một ảnh sai người tệ hơn hẳn việc không có ảnh.

### Benchmark chuẩn nếu cần so sánh học thuật

Paired trên VITON-HD test: SSIM, LPIPS, FID. Unpaired: FID/KID. Baseline nên so: IDM-VTON và CatVTON.

---

## 9. Hạn chế đã biết & cách khắc phục

| Hiện tượng | Nguyên nhân | Xử lý |
|---|---|---|
| Logo/chữ trên áo bị bịa sai | Kontext không copy pixel-exact | Pass 2: crop vùng ngực + ảnh close-up đồ làm reference, prompt `restore the exact printed graphic` |
| Áo "dán phẳng", không ôm dáng | Guidance cao, thiếu mô tả vật lý | Hạ guidance về 2.0, thêm `fabric wrapping around the torso with natural shadows` |
| Đổi luôn quần / giày / nền | Mệnh đề preservation chưa liệt kê đủ | Bổ sung danh sách cụ thể (mục 5) |
| Mặt "đẹp lên", nền lệch | Bản chất của việc sinh lại toàn khung | Mask compositing (mục 7) |
| Viền áo cũ còn sót | Mask dilate chưa đủ | Tăng `iterations` lên 6–8 |
| Kết quả dao động mạnh | Variance theo seed | Best-of-K |
| OOM | bf16 hoặc canvas quá lớn | Dùng fp8, thêm `--lowvram`, hoặc hạ về 512×704 mỗi nửa |

### Lộ trình lên chất lượng thương mại: LoRA

Kontext zero-shot đạt mức "demo tốt". Để lên mức sản phẩm, fine-tune LoRA trên dữ liệu ghép cặp:

- **Dữ liệu:** VITON-HD / DressCode, mỗi mẫu = `(canvas[người ∣ đồ] → ảnh người mặc đồ đó)`.
- **Cấu hình:** LoRA rank 32–64, lr 1e-4, 2–4k step, batch 1 + grad accum 4. Vừa 1× A100 40 GB, hoặc 24 GB nếu bật 8-bit optimizer + gradient checkpointing.
- **Lợi:** tăng mạnh độ trung thực texture/logo, giảm drift → có thể nới lỏng bước refine.
- Thêm LoRA vào graph: chèn `LoraLoaderModelOnly` giữa node 1 và node 15.

Lưu ý: fine-tune **không thay thế** được mục 7 và mục 8 — compositing và QA vẫn giữ nguyên.

### Giấy phép

`FLUX.1 Kontext [dev]` là **non-commercial license**. Sản phẩm thương mại phải dùng `[pro]`/`[max]` qua BFL API, hoặc mua commercial license. Quyết định này ảnh hưởng ngược lên kiến trúc (self-host vs API) nên cần chốt sớm. Bản `[pro]`/`[max]` cũng nhận nhiều ảnh trực tiếp, không cần ghép canvas.

---

## 10. Cách chạy

### Qua giao diện ComfyUI

1. Copy `person.png` và `garment.png` vào `ComfyUI/input/`.
2. Mở ComfyUI → **Workflow → Open** → chọn `flux_kontext_vton.json` (hoặc kéo-thả file vào canvas).
3. Kiểm tra dropdown của node 1, 2, 3, 19 khớp tên file model thực tế trên máy.
4. Sửa prompt ở node 11 theo món đồ.
5. **Queue Prompt**. Kết quả ở `ComfyUI/output/vton/`.

### Qua API

```bash
python - <<'P'
import json, urllib.request
g = json.load(open("flux_kontext_vton_api.json"))
g["4"]["inputs"]["image"] = "person.png"
g["5"]["inputs"]["image"] = "garment.png"
g["11"]["inputs"]["text"] = "Replace the top worn by the person on the left with ..."
g["15"]["inputs"]["seed"] = 12345
req = urllib.request.Request("http://127.0.0.1:8188/prompt",
                             json.dumps({"prompt": g}).encode(),
                             {"Content-Type": "application/json"})
print(urllib.request.urlopen(req).read().decode())
P
```

### Yêu cầu ảnh đầu vào

- **Người:** ảnh nửa người hoặc toàn thân, thấy rõ trang phục hiện tại, tư thế đứng thẳng, nền không quá rối.
- **Đồ:** ảnh phẳng (flat-lay) hoặc ghost-mannequin trên **nền trắng**, chụp chính diện. Ảnh đồ đang mặc trên người mẫu khác cho kết quả kém hơn hẳn — nếu buộc phải dùng thì nên tách nền và đặt lên nền trắng trước.

---

## 11. Cấu trúc thư mục

```
D:\vton-kontext\
├── README.md                      # báo cáo này
├── flux_kontext_vton.json         # workflow UI-format (kéo-thả vào ComfyUI)
├── flux_kontext_vton_api.json     # workflow API-format (POST /prompt)
└── bundle\
    └── remote_setup.sh            # script tải model + khởi động ComfyUI trên server
```
