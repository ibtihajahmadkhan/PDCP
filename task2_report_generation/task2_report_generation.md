# Task 2 – Medical Report Generation (Visual Language Model)

## 1. Objective

The objective of this task is to integrate an open-source **Vision-Language Model (VLM)** to generate a concise, radiology-style report from a PneumoniaMNIST chest X-ray image.

This task focuses on:

- Running inference with a pretrained medical VLM
- Designing and comparing prompting strategies
- Generating reports for ≥10 test samples
- Qualitative evaluation of generated text
- Cross-model comparison with the CNN classifier from Task 1

The goal is not to train a generative model, but to evaluate the behavior and reliability of an existing multimodal model under constrained medical imaging conditions.

---

## 2. Dataset & Constraints

**Dataset:** PneumoniaMNIST (MedMNIST v2)  
**Image Resolution:** 28×28 grayscale  
**Labels:**  
- 0 = Normal  
- 1 = Pneumonia  

### Critical Constraint

The extremely low resolution (28×28) severely limits visual information. Subtle radiographic findings such as small consolidations or effusions may not be visually discernible at this scale.

This constraint strongly influences:

- VLM grounding reliability  
- Hallucination behavior  
- Diagnostic confidence  

Therefore, evaluation must emphasize qualitative reasoning rather than assume radiologist-level certainty.

---

## 3. Model Selection

### Selected Model

**LLaVA-Med v1.5 (Mistral-7B), loaded in 4-bit quantization**

### Rationale

- Medical-domain adaptation (vision-language biomedical setting)
- Feasible deployment on laptop GPU via 4-bit quantization
- Appropriate for inference-focused evaluation

The model is used strictly for inference; no fine-tuning was performed.

---

## 4. Implementation Overview

### Inference Pipeline

1. Load pretrained VLM + processor  
2. Load PneumoniaMNIST test image  
3. Construct chat-style prompt including image token  
4. Generate report using deterministic decoding  
5. Save outputs for analysis  

### Files Produced

- Report generation script:  
  `task2_report_generation/generate_10_reports.py`

- Generated reports (10 samples × 2 prompt strategies):  
  `reports/task2/generated_reports.md`

- Saved test images:  
  `reports/task2/images/*.png`

- CNN vs VLM comparison table:  
  `reports/task2/cnn_vlm_comparison.md`

---

## 5. Prompting Strategies

Two prompting strategies were evaluated on the same 10 test images (5 pneumonia, 5 normal).

---

### 5.1 Prompt V1 – Simple Radiology Instruction

**Prompt:**

> Write a short chest X-ray report with sections FINDINGS and IMPRESSION. Be clinically grounded.

**Observed Behavior:**

- Frequently produces generic “unremarkable” impressions.
- Often under-calls pneumonia (GT=1).
- Appears conservative when uncertain.

---

### 5.2 Prompt V2 – Structured + Uncertainty + Constraints

This prompt included:

- Structured bullet points in FINDINGS  
- Required comments on consolidation, pleural effusion, pneumothorax, cardiac size  
- Explicit allowance of uncertainty due to low resolution  
- Instruction to avoid hallucinating devices or lines  

**Observed Behavior:**

- Produces more structured radiology-style text.
- Frequently asserts specific pathologies in normal cases.
- Demonstrates prompt-induced hallucination under forced attribute enumeration.

---

## 6. Sample Outputs

All generated reports are available in:

`reports/task2/generated_reports.md`

Each sample includes:

- Test image  
- Prompt used  
- Generated report  
- Test index  
- Ground-truth label  

---

## 7. Qualitative Evaluation & Cross-Model Comparison

### 7.1 Alignment with Ground Truth

Across 10 test samples:

- Prompt V1 often describes pneumonia cases as normal.
- Prompt V2 frequently hallucinates findings in normal cases.
- The VLM struggles to reliably ground pathology at 28×28 resolution.
- Output quality is highly sensitive to prompt design.

This highlights the instability of generative reasoning under low-information visual inputs.

---

### 7.2 CNN vs VLM Comparison

Comparison results are documented in:

`reports/task2/cnn_vlm_comparison.md`

#### CNN Behavior

The Task 1 ResNet18 classifier produces highly separable probabilities:

- GT=0 → probabilities ≈ 0.0000–0.0038  
- GT=1 → probabilities ≈ 0.9895–1.0000  

This indicates strong discrimination and calibration.

#### VLM Behavior

Prompt V1:
- Conservative, often defaults to normal findings.

Prompt V2:
- More structured, but prone to hallucinated pathology.
- Repeated mentions of pleural effusion in normal cases were observed.

---

### 7.3 Key Insight

| Model | Strength | Limitation |
|--------|----------|------------|
| CNN | High-confidence binary prediction | No narrative explanation |
| VLM | Produces structured reports | Prompt-sensitive; hallucination risk |

---

### 7.4 Hybrid Integration Strategy

A safer deployment design would:

1. Use CNN for primary pathology detection.
2. Use VLM for structured report generation.
3. Constrain VLM output using CNN confidence (consistency gating).

This hybrid strategy reduces hallucination risk while preserving interpretability.

---

## 8. Strengths & Limitations

### Strengths

- End-to-end VLM inference pipeline
- Multiple prompt strategies compared
- Cross-model comparison performed
- Transparent hallucination analysis
- Clinically cautious interpretation

### Limitations

- Extremely low input resolution (28×28)
- No ground-truth radiology reports available
- Evaluation remains qualitative
- Structured prompts can increase over-specification

---

## 9. Future Improvements

- CNN-conditioned prompting
- Explicit uncertainty calibration templates
- Evaluation on higher-resolution medical datasets
- Analysis on CNN misclassified cases

---

## 10. Reproducibility

To reproduce results:

```bash
python task2_report_generation/generate_10_reports.py
```

⚠ Task 2 requires CUDA-compatible environment for 4-bit quantization.
If installation fails, disable 4-bit loading in model.py.

In your Colab, put this cell before installing requirements:
```bash
!pip -q install --upgrade pip
!pip -q install -r requirements.txt
```
