import os
import random
import numpy as np
import torch
from PIL import Image

from medmnist import PneumoniaMNIST
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig


def load_vlm(model_id: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()
    return processor, model


def get_image_and_label(ds, idx: int):
    img, label = ds[idx]
    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.array(img))
    return img.convert("RGB"), int(label[0])


def build_prompt(processor, strict: bool):
    # Two prompting strategies to document (required by Task 2)
    if strict:
        user_text = (
            "You are a radiologist. Provide a concise chest X-ray report with:\n"
            "FINDINGS: 2-4 bullet points.\n"
            "IMPRESSION: 1-2 lines.\n"
            "If you are uncertain due to low resolution, explicitly state uncertainty.\n"
            "Do NOT hallucinate devices or lines.\n"
            "Comment specifically on: consolidation/infiltrate, pleural effusion, pneumothorax, cardiac size.\n"
        )
    else:
        user_text = (
            "Write a short chest X-ray report with sections FINDINGS and IMPRESSION.\n"
            "Be clinically grounded.\n"
        )

    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": user_text}],
        }
    ]
    return processor.apply_chat_template(messages, add_generation_prompt=True), user_text


@torch.inference_mode()
def generate_report(processor, model, image, prompt):
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=220,
        do_sample=False,
        num_beams=3,
        repetition_penalty=1.1,
    )
    decoded = processor.batch_decode(out, skip_special_tokens=True)[0]
    text = decoded[len(prompt):].strip() if decoded.startswith(prompt) else decoded.strip()
    return text


def pick_indices(ds, n_pos=5, n_neg=5, seed=42):
    rng = random.Random(seed)
    pos = [i for i in range(len(ds)) if int(ds[i][1][0]) == 1]
    neg = [i for i in range(len(ds)) if int(ds[i][1][0]) == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    return pos[:n_pos], neg[:n_neg]


def main():
    # HF-compatible LLaVA-Med checkpoint
    model_id = "chaoyinshe/llava-med-v1.5-mistral-7b-hf"

    os.makedirs("./reports/task2/images", exist_ok=True)
    os.makedirs("./reports/task2", exist_ok=True)

    processor, model = load_vlm(model_id)

    ds_test = PneumoniaMNIST(root="./data", split="test", download=True)

    pos_idx, neg_idx = pick_indices(ds_test, n_pos=5, n_neg=5, seed=42)

    # Alternate prompting strategies (document effects)
    strategies = [
        ("prompt_v1_simple", False),
        ("prompt_v2_strict_uncertainty", True),
    ]

    rows = []
    sample_list = [(i, 1) for i in pos_idx] + [(i, 0) for i in neg_idx]

    for k, (idx, gt) in enumerate(sample_list):
        image, gt_label = get_image_and_label(ds_test, idx)
        img_path = f"./reports/task2/images/sample_{k:02d}_idx{idx}_gt{gt_label}.png"
        image.save(img_path)

        for strat_name, strict in strategies:
            prompt, prompt_text = build_prompt(processor, strict=strict)
            report = generate_report(processor, model, image, prompt)

            rows.append({
                "sample_id": f"sample_{k:02d}",
                "test_index": idx,
                "gt_label": gt_label,  # 0 normal, 1 pneumonia
                "prompt_strategy": strat_name,
                "prompt_text": prompt_text,
                "image_path": img_path.replace("\\", "/"),
                "generated_report": report
            })

        print(f"Done sample {k:02d} (idx={idx}, gt={gt_label})")

    # Save as markdown for easy viewing
    md_path = "./reports/task2/generated_reports.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Task 2 – Sample Generated Reports (10 Images)\n\n")
        f.write("Model: `chaoyinshe/llava-med-v1.5-mistral-7b-hf` (4-bit)\n\n")
        f.write("Labels: 0=Normal, 1=Pneumonia\n\n")
        for r in rows:
            f.write(f"---\n\n")
            f.write(f"## {r['sample_id']} | test_idx={r['test_index']} | GT={r['gt_label']} | {r['prompt_strategy']}\n\n")
            f.write(f"![{r['sample_id']}]({r['image_path']})\n\n")
            f.write("**Prompt used:**\n\n")
            f.write("```text\n")
            f.write(r["prompt_text"].strip() + "\n")
            f.write("```\n\n")
            f.write("**Generated report:**\n\n")
            f.write("```text\n")
            f.write(r["generated_report"].strip() + "\n")
            f.write("```\n\n")

    print("\nSaved:", md_path)
    print("Images saved in: ./reports/task2/images/")


if __name__ == "__main__":
    main()
