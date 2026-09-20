#!/usr/bin/env python3
"""
build_comfy_workflow.py
Sinh workflow ComfyUI cho LoRA "sks_dog_lora" o CA HAI dinh dang:
  - GUI format  -> /workspace/ComfyUI/user/default/workflows/sks_dog_inference.json
  - API format  -> /workspace/workflows/sks_dog_inference_api.json

Thiet ke workflow (tai sao dung nhung node nay):
  TOC DO   : LCM-LoRA + sampler "lcm" + scheduler "sgm_uniform" + CFG thap
             -> 8 step thay vi 25-30 step (nhanh ~2.5x).
             CANH BAO: KHONG dung node ModelSamplingDiscrete(sampling="lcm")
             kem sampler "lcm" tren ComfyUI >= 0.35. Sampler "lcm" da tu xu ly
             buoc nhay LCM; them ModelSamplingDiscrete la ap dung 2 lan
             -> anh ra NHIEU TRANG (da kiem chung tren instance nay).
  CHAT LUONG: FreeU_V2 (tai can bang skip-connection cua UNet -> net hon,
             it "mu suong"), hires-fix 2 pass (512 -> 768 latent, denoise 0.45)
             de them chi tiet ma khong bi "double head", cuoi cung
             ESRGAN 4x-UltraSharp + ImageScale ve 1536px.
"""
import json
import os

CKPT = "v1-5-pruned-emaonly-fp16.safetensors"
SUBJECT_LORA = "sks_dog_lora.safetensors"
LCM_LORA = "lcm-lora-sdv1-5.safetensors"
UPSCALER = "4x-UltraSharp.pth"

POSITIVE = ("a professional photo of sks dog sitting in a garden full of flowers, "
            "golden hour sunlight, shallow depth of field, highly detailed fur, sharp focus")
NEGATIVE = "blurry, low quality, deformed, extra limbs, bad anatomy, watermark, text"

# (id, class_type, widgets_values, inputs{name: (from_node_id, output_slot)}, pos, title)
NODES = [
    (1,  "CheckpointLoaderSimple", [CKPT], {}, (40, 120), "Base SD1.5"),
    (2,  "LoraLoader", [SUBJECT_LORA, 0.75, 0.75],
         {"model": (1, 0), "clip": (1, 1)}, (350, 120), "LoRA: sks dog (da train)"),
    (3,  "LoraLoader", [LCM_LORA, 1.0, 1.0],
         {"model": (2, 0), "clip": (2, 1)}, (350, 300), "LCM-LoRA (tang toc)"),
    (4,  "FreeU_V2", [1.3, 1.4, 0.9, 0.2],
         {"model": (3, 0)}, (680, 120), "FreeU_V2 (tang chat luong)"),
    (6,  "CLIPTextEncode", [POSITIVE], {"clip": (3, 1)}, (680, 400), "Positive"),
    (7,  "CLIPTextEncode", [NEGATIVE], {"clip": (3, 1)}, (680, 600), "Negative"),
    (8,  "EmptyLatentImage", [512, 512, 1], {}, (680, 790), "Latent 512"),
    (9,  "KSampler", [42, "randomize", 8, 2.0, "lcm", "sgm_uniform", 1.0],
         {"model": (4, 0), "positive": (6, 0), "negative": (7, 0), "latent_image": (8, 0)},
         (1020, 120), "Pass 1 - LCM 8 steps"),
    (10, "LatentUpscale", ["nearest-exact", 768, 768, "disabled"],
         {"samples": (9, 0)}, (1020, 480), "Hires: latent 768"),
    (11, "KSampler", [42, "randomize", 6, 2.0, "lcm", "sgm_uniform", 0.45],
         {"model": (4, 0), "positive": (6, 0), "negative": (7, 0), "latent_image": (10, 0)},
         (1330, 120), "Pass 2 - hires fix"),
    (12, "VAEDecode", [], {"samples": (11, 0), "vae": (1, 2)}, (1330, 480), "VAE Decode"),
    (13, "UpscaleModelLoader", [UPSCALER], {}, (1330, 600), "4x-UltraSharp"),
    (14, "ImageUpscaleWithModel", [],
         {"upscale_model": (13, 0), "image": (12, 0)}, (1620, 480), "ESRGAN 4x"),
    (15, "ImageScale", ["lanczos", 1536, 1536, "disabled"],
         {"image": (14, 0)}, (1620, 620), "Resize 1536"),
    (16, "SaveImage", ["sks_dog"], {"images": (15, 0)}, (1900, 120), "Save"),
]

# Thu tu socket input cua tung class (theo object_info) - de map dung slot index
INPUT_ORDER = {
    "LoraLoader": ["model", "clip"],
    "FreeU_V2": ["model"],
    "ModelSamplingDiscrete": ["model"],
    "CLIPTextEncode": ["clip"],
    "KSampler": ["model", "positive", "negative", "latent_image"],
    "LatentUpscale": ["samples"],
    "VAEDecode": ["samples", "vae"],
    "ImageUpscaleWithModel": ["upscale_model", "image"],
    "ImageScale": ["image"],
    "SaveImage": ["images"],
}
OUTPUTS = {
    "CheckpointLoaderSimple": ["MODEL", "CLIP", "VAE"],
    "LoraLoader": ["MODEL", "CLIP"],
    "FreeU_V2": ["MODEL"],
    "ModelSamplingDiscrete": ["MODEL"],
    "CLIPTextEncode": ["CONDITIONING"],
    "EmptyLatentImage": ["LATENT"],
    "KSampler": ["LATENT"],
    "LatentUpscale": ["LATENT"],
    "VAEDecode": ["IMAGE"],
    "UpscaleModelLoader": ["UPSCALE_MODEL"],
    "ImageUpscaleWithModel": ["IMAGE"],
    "ImageScale": ["IMAGE"],
    "SaveImage": [],
}
# Ten widget theo thu tu, dung cho API format (KSampler co them 'control_after_generate'
# chi ton tai o GUI -> phai bo khi xuat API format)
WIDGETS = {
    "CheckpointLoaderSimple": ["ckpt_name"],
    "LoraLoader": ["lora_name", "strength_model", "strength_clip"],
    "FreeU_V2": ["b1", "b2", "s1", "s2"],
    "ModelSamplingDiscrete": ["sampling", "zsnr"],
    "CLIPTextEncode": ["text"],
    "EmptyLatentImage": ["width", "height", "batch_size"],
    "KSampler": ["seed", "__control__", "steps", "cfg", "sampler_name", "scheduler", "denoise"],
    "LatentUpscale": ["upscale_method", "width", "height", "crop"],
    "VAEDecode": [],
    "UpscaleModelLoader": ["model_name"],
    "ImageUpscaleWithModel": [],
    "ImageScale": ["upscale_method", "width", "height", "crop"],
    "SaveImage": ["filename_prefix"],
}


def build_api():
    graph = {}
    for nid, cls, widgets, inputs, _pos, title in NODES:
        node_inputs = {}
        for name, (src, slot) in inputs.items():
            node_inputs[name] = [str(src), slot]
        for wname, wval in zip(WIDGETS[cls], widgets):
            if wname == "__control__":
                continue
            node_inputs[wname] = wval
        graph[str(nid)] = {
            "class_type": cls,
            "inputs": node_inputs,
            "_meta": {"title": title},
        }
    return graph


def build_gui():
    link_id = 0
    links = []
    out_links = {}   # (node, slot) -> [link ids]
    in_link = {}     # (node, input_name) -> link id
    types = {}
    for nid, cls, *_ in NODES:
        types[nid] = cls

    for nid, cls, widgets, inputs, _pos, _title in NODES:
        for name in INPUT_ORDER.get(cls, []):
            if name not in inputs:
                continue
            src, slot = inputs[name]
            link_id += 1
            ltype = OUTPUTS[types[src]][slot]
            links.append([link_id, src, slot, nid, INPUT_ORDER[cls].index(name), ltype])
            out_links.setdefault((src, slot), []).append(link_id)
            in_link[(nid, name)] = link_id

    nodes = []
    for order, (nid, cls, widgets, inputs, pos, title) in enumerate(NODES):
        node_inputs = []
        for idx, name in enumerate(INPUT_ORDER.get(cls, [])):
            if name not in inputs:
                continue
            src, slot = inputs[name]
            node_inputs.append({
                "name": name,
                "type": OUTPUTS[types[src]][slot],
                "link": in_link[(nid, name)],
            })
        node_outputs = []
        for slot, otype in enumerate(OUTPUTS[cls]):
            node_outputs.append({
                "name": otype,
                "type": otype,
                "slot_index": slot,
                "links": out_links.get((nid, slot), []),
            })
        nodes.append({
            "id": nid,
            "type": cls,
            "pos": list(pos),
            "size": [330, 120],
            "flags": {},
            "order": order,
            "mode": 0,
            "inputs": node_inputs,
            "outputs": node_outputs,
            "title": title,
            "properties": {"Node name for S&R": cls},
            "widgets_values": list(widgets),
        })

    return {
        "id": "sks-dog-lora-inference",
        "revision": 0,
        "last_node_id": max(n[0] for n in NODES),
        "last_link_id": link_id,
        "nodes": nodes,
        "links": links,
        "groups": [],
        "config": {},
        "extra": {"ds": {"scale": 0.7, "offset": [0, 0]}},
        "version": 0.4,
    }


def main():
    api = build_api()
    gui = build_gui()

    os.makedirs("/workspace/workflows", exist_ok=True)
    gui_dir = "/workspace/ComfyUI/user/default/workflows"
    os.makedirs(gui_dir, exist_ok=True)

    api_path = "/workspace/workflows/sks_dog_inference_api.json"
    gui_path = os.path.join(gui_dir, "sks_dog_inference.json")
    with open(api_path, "w") as f:
        json.dump(api, f, indent=2)
    with open(gui_path, "w") as f:
        json.dump(gui, f, indent=2)

    print(f"API format : {api_path}")
    print(f"GUI format : {gui_path}  (mo trong ComfyUI: Workflows -> sks_dog_inference)")


if __name__ == "__main__":
    main()
