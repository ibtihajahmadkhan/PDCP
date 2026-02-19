import torch
from PIL import Image
import numpy as np

from medmnist import PneumoniaMNIST
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
import torch

def get_one_image(split="test", idx=0):
    ds = PneumoniaMNIST(root="./data", split=split, download=True)
    img, label = ds[idx]  # img is PIL Image, label is numpy array shape (1,)
    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.array(img))
    return img.convert("RGB"), int(label[0])


def main():
    model_id = "chaoyinshe/llava-med-v1.5-mistral-7b-hf"

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

    image, gt_label = get_one_image(split="test", idx=0)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                        "You are a radiologist. Write a concise chest X-ray report with:\n"
                        "FINDINGS: (2-4 bullet points)\n"
                        "IMPRESSION: (1-2 lines)\n"
                        "Be clinically grounded. If normal, clearly state no acute cardiopulmonary abnormality."
                    ),
                },
            ],
        }
    ]

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=220,
            do_sample=False,
            num_beams=3,
            repetition_penalty=1.1,
        )

    decoded = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    text = decoded[len(prompt):].strip() if decoded.startswith(prompt) else decoded.strip()

    print("\n=== Ground Truth Label (PneumoniaMNIST) ===")
    print("Label:", gt_label, "(0=Normal, 1=Pneumonia)")
    print("\n=== Generated Report ===")
    print(text)


if __name__ == "__main__":
    main()
