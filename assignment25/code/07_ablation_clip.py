#!/usr/bin/env python3
"""
07_ablation_clip.py
Chay 04 + 05 voi nhieu LORA_STRENGTH khac nhau de co BANG SO SANH
CLIP-I / CLIP-T cho bao cao (trade-off subject fidelity <-> prompt fidelity).

Chay:
    source /workspace/sd-scripts/venv/bin/activate
    python3 07_ablation_clip.py
Ket qua: /workspace/lora_train/eval/ablation_clip.json  (+ bang in ra man hinh)
"""
import json
import os
import shutil
import subprocess
import sys

WS = "/workspace"
VENV_PY = os.path.join(WS, "sd-scripts", "venv", "bin", "python3")
EVAL_DIR = os.path.join(WS, "lora_train", "eval")
GEN_DIR = os.path.join(EVAL_DIR, "generated")
SCORES = os.path.join(EVAL_DIR, "clip_scores.json")
RESULT = os.path.join(EVAL_DIR, "ablation_clip.json")

STRENGTHS = [0.6, 0.8, 1.0]


def main():
    rows = []
    for s in STRENGTHS:
        print(f"\n=========== LORA_STRENGTH = {s} ===========")
        env = dict(os.environ, LORA_STRENGTH=str(s))
        subprocess.run([VENV_PY, os.path.join(WS, "04_generate_samples.py")],
                       cwd=WS, env=env, check=True)
        subprocess.run([VENV_PY, os.path.join(WS, "05_evaluate_clip.py")],
                       cwd=WS, env=env, check=True)

        with open(SCORES) as f:
            r = json.load(f)
        rows.append({
            "lora_strength": s,
            "CLIP-I": r["CLIP-I_overall"],
            "CLIP-T": r["CLIP-T_overall"],
        })
        # giu lai anh cua tung muc strength de doi chieu bang mat
        keep = os.path.join(EVAL_DIR, f"generated_s{str(s).replace('.', '')}")
        if os.path.exists(keep):
            shutil.rmtree(keep)
        shutil.copytree(GEN_DIR, keep)
        shutil.copy(SCORES, os.path.join(EVAL_DIR,
                                         f"clip_scores_s{str(s).replace('.', '')}.json"))

    with open(RESULT, "w") as f:
        json.dump(rows, f, indent=2)

    print("")
    print("==========================================")
    print(f"{'LoRA strength':>14} | {'CLIP-I':>7} | {'CLIP-T':>7}")
    print("-" * 36)
    for r in rows:
        print(f"{r['lora_strength']:>14} | {r['CLIP-I']:>7.4f} | {r['CLIP-T']:>7.4f}")
    print("==========================================")
    print(f"  Luu tai: {RESULT}")
    print("  CLIP-I tang / CLIP-T giam khi strength tang = trade-off dien hinh.")


if __name__ == "__main__":
    main()
