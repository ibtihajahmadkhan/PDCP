import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import re
import math
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from medmnist import PneumoniaMNIST

# Import your Task-1 model (make sure this path is correct)
# This assumes you have task1_classification/model.py with ResNet18_Gray class
from task1_classification.model import ResNet18_Gray


# ---- Adjust these if your filenames differ ----
TASK2_MD_PATH = "./reports/task2/generated_reports.md"
CNN_CKPT_PATH = "./models/best_resnet18.pt"   # <- your Task-1 best checkpoint
OUT_MD_PATH   = "./reports/task2/cnn_vlm_comparison.md"

# If you tuned a balanced-accuracy threshold, set it here (optional).
# If unknown, keep 0.5 and still report probabilities.
CNN_THRESHOLD = 0.5


def load_task1_model(device: str):
    model = ResNet18_Gray(pretrained=False).to(device)
    ckpt = torch.load(CNN_CKPT_PATH, map_location=device, weights_only=True)

    # Handle common checkpoint formats
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def get_eval_transform(img_size=224):
    # Match your Task-1 eval preprocessing
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),                  # -> [1,H,W] for grayscale
        transforms.Normalize(mean=[0.485], std=[0.229]),
    ])


@torch.inference_mode()
def predict_cnn_prob(model, pil_img, tfm, device: str):
    x = tfm(pil_img).unsqueeze(0).to(device)  # [1,1,224,224]
    logits = model(x).view(-1)
    prob = torch.sigmoid(logits).item()
    pred = 1 if prob >= CNN_THRESHOLD else 0
    return prob, pred


def extract_impression(text: str, max_len=160) -> str:
    """
    Extract a short impression snippet from a generated report.
    Robust to LLaVA/Mistral [INST] blocks and different formatting.
    """
    t = text.strip()

    # Remove instruction wrapper if present
    # Keep only content AFTER [/INST] when available
    if "[/INST]" in t:
        t = t.split("[/INST]", 1)[1].strip()

    # Normalize whitespace for easier regex
    t_norm = t.replace("\r\n", "\n").replace("\r", "\n")

    # 1) Try to capture the IMPRESSION section content
    # Grab text after "IMPRESSION:" until next all-caps header or end.
    m = re.search(r"IMPRESSION\s*:\s*(.*)", t_norm, flags=re.IGNORECASE)
    if m:
        tail = t_norm[m.end():].strip()

        # Take first 1-2 non-empty lines
        lines = [ln.strip("-• \t") for ln in tail.splitlines() if ln.strip()]
        if lines:
            snippet = lines[0]
            if len(lines) > 1 and len(snippet) < 80:
                snippet = snippet + " " + lines[1]
            snippet = re.sub(r"\s+", " ", snippet).strip()
            return snippet[:max_len]

    # 2) Fallback: take first line after FINDINGS if impression missing
    m2 = re.search(r"FINDINGS\s*:\s*(.*)", t_norm, flags=re.IGNORECASE)
    if m2:
        tail = t_norm[m2.end():].strip()
        lines = [ln.strip("-• \t") for ln in tail.splitlines() if ln.strip()]
        if lines:
            snippet = re.sub(r"\s+", " ", lines[0]).strip()
            return snippet[:max_len]

    # 3) Last fallback: first ~max_len chars
    t_flat = re.sub(r"\s+", " ", t)
    return t_flat[:max_len]



def parse_task2_md(md_path: str):
    """
    Parse generated_reports.md produced by generate_10_reports.py.
    Returns:
      samples: dict[test_idx] = {
         "gt": int,
         "sample_id": str,
         "v1": short_impression,
         "v2": short_impression,
         "v1_full": full_report,
         "v2_full": full_report
      }
    """
    with open(md_path, "r", encoding="utf-8") as f:
        md = f.read()

    # Each block starts with:
    # ## sample_00 | test_idx=555 | GT=1 | prompt_v1_simple
    header_re = re.compile(
        r"^##\s+(?P<sample>sample_\d+)\s+\|\s+test_idx=(?P<idx>\d+)\s+\|\s+GT=(?P<gt>\d+)\s+\|\s+(?P<strategy>[^\n]+)\s*$",
        re.MULTILINE
    )

    # report is inside the last ```text ... ``` in the block
    # We'll capture everything between "**Generated report:**" and the next "```"
    report_re = re.compile(
        r"\*\*Generated report:\*\*\s*```text\s*(?P<report>.*?)\s*```",
        re.DOTALL | re.IGNORECASE
    )

    headers = list(header_re.finditer(md))
    samples = {}

    for i, h in enumerate(headers):
        start = h.start()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(md)
        block = md[start:end]

        sample_id = h.group("sample")
        idx = int(h.group("idx"))
        gt = int(h.group("gt"))
        strategy = h.group("strategy").strip()

        rm = report_re.search(block)
        report_text = rm.group("report").strip() if rm else ""

        imp = extract_impression(report_text)

        if idx not in samples:
            samples[idx] = {"gt": gt, "sample_id": sample_id}

        # Store strategy results
        if "prompt_v1" in strategy:
            samples[idx]["v1"] = imp
            samples[idx]["v1_full"] = report_text
        elif "prompt_v2" in strategy:
            samples[idx]["v2"] = imp
            samples[idx]["v2_full"] = report_text
        else:
            # fallback bucket
            samples[idx].setdefault("other", []).append((strategy, imp))

    return samples


def main():
    assert os.path.exists(TASK2_MD_PATH), f"Missing: {TASK2_MD_PATH}"
    assert os.path.exists(CNN_CKPT_PATH), f"Missing: {CNN_CKPT_PATH}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    samples = parse_task2_md(TASK2_MD_PATH)
    if not samples:
        raise RuntimeError("No samples parsed from generated_reports.md")

    # Load dataset and model
    ds_test = PneumoniaMNIST(root="./data", split="test", download=True)
    tfm = get_eval_transform(img_size=224)
    model = load_task1_model(device)

    # Prepare rows
    rows = []
    for test_idx in sorted(samples.keys()):
        pil_img, gt = ds_test[test_idx]
        gt = int(gt[0])

        prob, pred = predict_cnn_prob(model, pil_img, tfm, device)
        row = {
            "test_idx": test_idx,
            "GT": gt,
            "CNN_prob(pneumonia)": prob,
            "CNN_pred": pred,
            "VLM_v1_impression": samples[test_idx].get("v1", ""),
            "VLM_v2_impression": samples[test_idx].get("v2", ""),
        }
        rows.append(row)

    os.makedirs(os.path.dirname(OUT_MD_PATH), exist_ok=True)

    # Write Markdown
    with open(OUT_MD_PATH, "w", encoding="utf-8") as f:
        f.write("# Task 2 – CNN vs VLM Comparison (10 Samples)\n\n")
        f.write(f"- CNN checkpoint: `{CNN_CKPT_PATH}`\n")
        f.write(f"- CNN threshold: `{CNN_THRESHOLD}`\n")
        f.write(f"- Source VLM outputs: `{TASK2_MD_PATH}`\n\n")

        f.write("| test_idx | GT | CNN_prob(pneumonia) | CNN_pred | VLM v1 (IMPRESSION snippet) | VLM v2 (IMPRESSION snippet) |\n")
        f.write("|---:|---:|---:|---:|---|---|\n")

        for r in rows:
            v1 = r["VLM_v1_impression"].replace("|", "/")
            v2 = r["VLM_v2_impression"].replace("|", "/")

            # truncate for readability
            v1 = (v1[:120] + "…") if len(v1) > 120 else v1
            v2 = (v2[:120] + "…") if len(v2) > 120 else v2

            f.write(
                f"| {r['test_idx']} | {r['GT']} | {r['CNN_prob(pneumonia)']:.4f} | {r['CNN_pred']} | {v1} | {v2} |\n"
            )

    print("Saved:", OUT_MD_PATH)
    print("\nPreview (first 3 rows):")
    for r in rows[:3]:
        print(r)


if __name__ == "__main__":
    main()
