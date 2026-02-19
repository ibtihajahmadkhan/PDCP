# Task 2 – Sample Generated Reports (10 Images)

Model: `chaoyinshe/llava-med-v1.5-mistral-7b-hf` (4-bit)

Labels: 0=Normal, 1=Pneumonia

---

## sample_00 | test_idx=555 | GT=1 | prompt_v1_simple

![sample_00](./reports/task2/images/sample_00_idx555_gt1.png)

**Prompt used:**

```text
Write a short chest X-ray report with sections FINDINGS and IMPRESSION.
Be clinically grounded.
```

**Generated report:**

```text
[INST] 
Write a short chest X-ray report with sections FINDINGS and IMPRESSION.
Be clinically grounded.
 [/INST] FINDINGS: The chest X-ray appears to be unremarkable, meaning that there are no significant abnormalities or findings detected in the image.

IMPRESSION: Based on the unremarkable chest X-ray, it seems that there are no major issues or concerns related to the patient's lungs or chest area. However, it is important to consider the patient's clinical history and symptoms, as well as consult a healthcare professional for a thorough evaluation and proper diagnosis of any underlying conditions that may not be visible on the X-ray.
```

---

## sample_00 | test_idx=555 | GT=1 | prompt_v2_strict_uncertainty

![sample_00](./reports/task2/images/sample_00_idx555_gt1.png)

**Prompt used:**

```text
You are a radiologist. Provide a concise chest X-ray report with:
FINDINGS: 2-4 bullet points.
IMPRESSION: 1-2 lines.
If you are uncertain due to low resolution, explicitly state uncertainty.
Do NOT hallucinate devices or lines.
Comment specifically on: consolidation/infiltrate, pleural effusion, pneumothorax, cardiac size.
```

**Generated report:**

```text
[INST] 
You are a radiologist. Provide a concise chest X-ray report with:
FINDINGS: 2-4 bullet points.
IMPRESSION: 1-2 lines.
If you are uncertain due to low resolution, explicitly state uncertainty.
Do NOT hallucinate devices or lines.
Comment specifically on: consolidation/infiltrate, pleural effusion, pneumothorax, cardiac size.
 [/INST] FINDINGS:
1. Consolidation/infiltrate in the right lower lobe.
2. Pleural effusion in the right lower lobe.
3. Pneumothorax in the right lower lobe.
4. Cardiac size is normal.

IMPRESSION:
The chest X-ray shows consolidation/infiltrate, pleural effusion, and pneumothorax in the right lower lobe. These findings suggest that there may be an underlying issue, such as an infection, inflammation, or other lung abnormality. Further evaluation and clinical correlation are needed to determine the cause and appropriate treatment.
```

---

## sample_01 | test_idx=495 | GT=1 | prompt_v1_simple

![sample_01](./reports/task2/images/sample_01_idx495_gt1.png)

**Prompt used:**

```text
Write a short chest X-ray report with sections FINDINGS and IMPRESSION.
Be clinically grounded.
```

**Generated report:**

```text
[INST] 
Write a short chest X-ray report with sections FINDINGS and IMPRESSION.
Be clinically grounded.
 [/INST] FINDINGS: The chest X-ray appears to be unremarkable, which means that there are no significant abnormalities or findings detected in the image.

IMPRESSION: Based on the unremarkable chest X-ray, it seems that there are no major issues or concerns related to the patient's lungs or chest area. However, it is important to consider the patient's clinical history and symptoms, as well as consult a healthcare professional for a thorough evaluation and proper diagnosis of any underlying conditions that may not be visible on the X-ray.
```

---

## sample_01 | test_idx=495 | GT=1 | prompt_v2_strict_uncertainty

![sample_01](./reports/task2/images/sample_01_idx495_gt1.png)

**Prompt used:**

```text
You are a radiologist. Provide a concise chest X-ray report with:
FINDINGS: 2-4 bullet points.
IMPRESSION: 1-2 lines.
If you are uncertain due to low resolution, explicitly state uncertainty.
Do NOT hallucinate devices or lines.
Comment specifically on: consolidation/infiltrate, pleural effusion, pneumothorax, cardiac size.
```

**Generated report:**

```text
[INST] 
You are a radiologist. Provide a concise chest X-ray report with:
FINDINGS: 2-4 bullet points.
IMPRESSION: 1-2 lines.
If you are uncertain due to low resolution, explicitly state uncertainty.
Do NOT hallucinate devices or lines.
Comment specifically on: consolidation/infiltrate, pleural effusion, pneumothorax, cardiac size.
 [/INST] FINDINGS:
1. Consolidation/infiltrate in the right lower lobe.
2. Pleural effusion.
3. Pneumothorax.
4. Cardiac size.

IMPRESSION:
The chest X-ray shows consolidation/infiltrate in the right lower lobe, pleural effusion, pneumothorax, and an enlarged cardiac silhouette. These findings suggest that there may be an underlying issue affecting the patient's respiratory and cardiovascular systems. Further evaluation and consultation with a healthcare professional are necessary to determine the cause and appropriate treatment for these findings.
```

---

## sample_02 | test_idx=573 | GT=1 | prompt_v1_simple

![sample_02](./reports/task2/images/sample_02_idx573_gt1.png)

**Prompt used:**

```text
Write a short chest X-ray report with sections FINDINGS and IMPRESSION.
Be clinically grounded.
```

**Generated report:**

```text
[INST] 
Write a short chest X-ray report with sections FINDINGS and IMPRESSION.
Be clinically grounded.
 [/INST] FINDINGS: The chest X-ray appears to be unremarkable, which means that there are no significant abnormalities or findings detected in the image.

IMPRESSION: Based on the unremarkable chest X-ray, it seems that there are no major issues or concerns related to the patient's lungs or chest area. However, it is important to consider the patient's clinical history and symptoms, as well as consult a healthcare professional for a thorough evaluation and proper diagnosis of any underlying conditions that may not be visible on the X-ray.
```

---

## sample_02 | test_idx=573 | GT=1 | prompt_v2_strict_uncertainty

![sample_02](./reports/task2/images/sample_02_idx573_gt1.png)

**Prompt used:**

```text
You are a radiologist. Provide a concise chest X-ray report with:
FINDINGS: 2-4 bullet points.
IMPRESSION: 1-2 lines.
If you are uncertain due to low resolution, explicitly state uncertainty.
Do NOT hallucinate devices or lines.
Comment specifically on: consolidation/infiltrate, pleural effusion, pneumothorax, cardiac size.
```

**Generated report:**

```text
[INST] 
You are a radiologist. Provide a concise chest X-ray report with:
FINDINGS: 2-4 bullet points.
IMPRESSION: 1-2 lines.
If you are uncertain due to low resolution, explicitly state uncertainty.
Do NOT hallucinate devices or lines.
Comment specifically on: consolidation/infiltrate, pleural effusion, pneumothorax, cardiac size.
 [/INST] FINDINGS:
1. Consolidation/infiltrate in the right lower lobe.
2. Pleural effusion.
3. Pneumothorax.
4. Cardiac size.

IMPRESSION:
The chest X-ray shows consolidation/infiltrate in the right lower lobe, pleural effusion, pneumothorax, and an enlarged cardiac silhouette. These findings suggest that there may be an underlying issue affecting the patient's respiratory and cardiovascular systems. Further evaluation and consultation with a healthcare professional are necessary to determine the cause and appropriate treatment for these findings.
```

---

## sample_03 | test_idx=393 | GT=1 | prompt_v1_simple

![sample_03](./reports/task2/images/sample_03_idx393_gt1.png)

**Prompt used:**

```text
Write a short chest X-ray report with sections FINDINGS and IMPRESSION.
Be clinically grounded.
```

**Generated report:**

```text
[INST] 
Write a short chest X-ray report with sections FINDINGS and IMPRESSION.
Be clinically grounded.
 [/INST] FINDINGS: The chest X-ray appears to be unremarkable, which means that there are no significant abnormalities or findings detected in the image.

IMPRESSION: Based on the unremarkable chest X-ray, it suggests that the patient's lungs and surrounding structures appear to be within normal limits. However, it is important to consider the patient's clinical history and symptoms, as well as consult a healthcare professional for a thorough evaluation and proper diagnosis of any underlying issues that may not be visible on the X-ray.
```

---

## sample_03 | test_idx=393 | GT=1 | prompt_v2_strict_uncertainty

![sample_03](./reports/task2/images/sample_03_idx393_gt1.png)

**Prompt used:**

```text
You are a radiologist. Provide a concise chest X-ray report with:
FINDINGS: 2-4 bullet points.
IMPRESSION: 1-2 lines.
If you are uncertain due to low resolution, explicitly state uncertainty.
Do NOT hallucinate devices or lines.
Comment specifically on: consolidation/infiltrate, pleural effusion, pneumothorax, cardiac size.
```

**Generated report:**

```text
[INST] 
You are a radiologist. Provide a concise chest X-ray report with:
FINDINGS: 2-4 bullet points.
IMPRESSION: 1-2 lines.
If you are uncertain due to low resolution, explicitly state uncertainty.
Do NOT hallucinate devices or lines.
Comment specifically on: consolidation/infiltrate, pleural effusion, pneumothorax, cardiac size.
 [/INST] FINDINGS:
1. Consolidation/infiltrate in the right lower lobe.
2. Pleural effusion.
3. Pneumothorax.
4. Cardiac size.

IMPRESSION:
The chest X-ray shows consolidation/infiltrate in the right lower lobe, pleural effusion, pneumothorax, and an enlarged cardiac silhouette. These findings suggest that there may be an underlying issue affecting the patient's respiratory and cardiovascular systems. Further evaluation and consultation with a healthcare professional are necessary to determine the cause and appropriate treatment for these findings.
```

---

## sample_04 | test_idx=124 | GT=1 | prompt_v1_simple

![sample_04](./reports/task2/images/sample_04_idx124_gt1.png)

**Prompt used:**

```text
Write a short chest X-ray report with sections FINDINGS and IMPRESSION.
Be clinically grounded.
```

**Generated report:**

```text
[INST] 
Write a short chest X-ray report with sections FINDINGS and IMPRESSION.
Be clinically grounded.
 [/INST] FINDINGS: The chest X-ray appears to be unremarkable, which means that there are no significant abnormalities or findings detected in the image.

IMPRESSION: Based on the unremarkable findings in the chest X-ray, it suggests that the patient's lungs and surrounding structures appear to be within normal limits at the time the image was taken. However, it is important to consider the patient's clinical history and symptoms, as well as consult a healthcare professional for a thorough evaluation and proper diagnosis of any underlying issues that may not be visible on the X-ray.
```

---

## sample_04 | test_idx=124 | GT=1 | prompt_v2_strict_uncertainty

![sample_04](./reports/task2/images/sample_04_idx124_gt1.png)

**Prompt used:**

```text
You are a radiologist. Provide a concise chest X-ray report with:
FINDINGS: 2-4 bullet points.
IMPRESSION: 1-2 lines.
If you are uncertain due to low resolution, explicitly state uncertainty.
Do NOT hallucinate devices or lines.
Comment specifically on: consolidation/infiltrate, pleural effusion, pneumothorax, cardiac size.
```

**Generated report:**

```text
[INST] 
You are a radiologist. Provide a concise chest X-ray report with:
FINDINGS: 2-4 bullet points.
IMPRESSION: 1-2 lines.
If you are uncertain due to low resolution, explicitly state uncertainty.
Do NOT hallucinate devices or lines.
Comment specifically on: consolidation/infiltrate, pleural effusion, pneumothorax, cardiac size.
 [/INST] FINDINGS:
1. Consolidation/infiltrate in the right lower lobe.
2. Pleural effusion in the right lower lobe.
3. Pneumothorax in the right lower lobe.
4. Cardiac size is normal.

IMPRESSION:
These findings suggest that there are abnormalities in the right lower lobe of the lung, including consolidation/infiltrate, pleural effusion, and pneumothorax. It is important to consider the patient's clinical history and symptoms, as well as consult a healthcare professional for a thorough evaluation and proper diagnosis of the underlying cause of these findings.
```

---

## sample_05 | test_idx=112 | GT=0 | prompt_v1_simple

![sample_05](./reports/task2/images/sample_05_idx112_gt0.png)

**Prompt used:**

```text
Write a short chest X-ray report with sections FINDINGS and IMPRESSION.
Be clinically grounded.
```

**Generated report:**

```text
[INST] 
Write a short chest X-ray report with sections FINDINGS and IMPRESSION.
Be clinically grounded.
 [/INST] FINDINGS: The chest X-ray appears to be unremarkable, which means that there are no significant abnormalities or findings detected in the image.

IMPRESSION: Based on the unremarkable chest X-ray, it seems that there are no major issues or concerns related to the patient's lungs or chest area. However, it is important to consider the patient's clinical history and symptoms, as well as consult a healthcare professional for a thorough evaluation and proper diagnosis of any underlying conditions that may not be visible on the X-ray.
```

---

## sample_05 | test_idx=112 | GT=0 | prompt_v2_strict_uncertainty

![sample_05](./reports/task2/images/sample_05_idx112_gt0.png)

**Prompt used:**

```text
You are a radiologist. Provide a concise chest X-ray report with:
FINDINGS: 2-4 bullet points.
IMPRESSION: 1-2 lines.
If you are uncertain due to low resolution, explicitly state uncertainty.
Do NOT hallucinate devices or lines.
Comment specifically on: consolidation/infiltrate, pleural effusion, pneumothorax, cardiac size.
```

**Generated report:**

```text
[INST] 
You are a radiologist. Provide a concise chest X-ray report with:
FINDINGS: 2-4 bullet points.
IMPRESSION: 1-2 lines.
If you are uncertain due to low resolution, explicitly state uncertainty.
Do NOT hallucinate devices or lines.
Comment specifically on: consolidation/infiltrate, pleural effusion, pneumothorax, cardiac size.
 [/INST] FINDINGS:
1. Consolidation/infiltrate in the right lower lobe.
2. Pleural effusion in the right lower lobe.
3. Pneumothorax in the right lower lobe.
4. Cardiac size is normal.

IMPRESSION:
The chest X-ray shows consolidation/infiltrate, pleural effusion, and pneumothorax in the right lower lobe. These findings suggest that there may be an underlying issue, such as an infection, inflammation, or other lung abnormality. Further evaluation and clinical correlation are needed to determine the cause and appropriate treatment.
```

---

## sample_06 | test_idx=145 | GT=0 | prompt_v1_simple

![sample_06](./reports/task2/images/sample_06_idx145_gt0.png)

**Prompt used:**

```text
Write a short chest X-ray report with sections FINDINGS and IMPRESSION.
Be clinically grounded.
```

**Generated report:**

```text
[INST] 
Write a short chest X-ray report with sections FINDINGS and IMPRESSION.
Be clinically grounded.
 [/INST] FINDINGS: The chest X-ray appears to be normal, with no significant abnormalities detected.

IMPRESSION: Based on the normal chest X-ray, the patient's condition seems to be stable. However, it is important to consider the patient's clinical history and symptoms, as well as consult a healthcare professional for a thorough evaluation and proper diagnosis of any underlying issues that may not be visible on the X-ray.
```

---

## sample_06 | test_idx=145 | GT=0 | prompt_v2_strict_uncertainty

![sample_06](./reports/task2/images/sample_06_idx145_gt0.png)

**Prompt used:**

```text
You are a radiologist. Provide a concise chest X-ray report with:
FINDINGS: 2-4 bullet points.
IMPRESSION: 1-2 lines.
If you are uncertain due to low resolution, explicitly state uncertainty.
Do NOT hallucinate devices or lines.
Comment specifically on: consolidation/infiltrate, pleural effusion, pneumothorax, cardiac size.
```

**Generated report:**

```text
[INST] 
You are a radiologist. Provide a concise chest X-ray report with:
FINDINGS: 2-4 bullet points.
IMPRESSION: 1-2 lines.
If you are uncertain due to low resolution, explicitly state uncertainty.
Do NOT hallucinate devices or lines.
Comment specifically on: consolidation/infiltrate, pleural effusion, pneumothorax, cardiac size.
 [/INST] FINDINGS:
1. Consolidation/infiltrate in the right lower lobe.
2. Pleural effusion in the right lower lobe.
3. Pneumothorax in the right lower lobe.
4. Cardiac size is normal.

IMPRESSION:
These findings suggest that there are abnormalities in the right lower lobe of the lung, including consolidation/infiltrate, pleural effusion, and pneumothorax. It is important to consider the patient's clinical history and symptoms, as well as consult a healthcare professional for a thorough evaluation and proper diagnosis of the underlying cause of these findings.
```

---

## sample_07 | test_idx=357 | GT=0 | prompt_v1_simple

![sample_07](./reports/task2/images/sample_07_idx357_gt0.png)

**Prompt used:**

```text
Write a short chest X-ray report with sections FINDINGS and IMPRESSION.
Be clinically grounded.
```

**Generated report:**

```text
[INST] 
Write a short chest X-ray report with sections FINDINGS and IMPRESSION.
Be clinically grounded.
 [/INST] FINDINGS: The chest X-ray appears to be unremarkable, which means that there are no significant abnormalities or findings detected in the image.

IMPRESSION: Based on the unremarkable chest X-ray, it seems that there are no major issues or concerns related to the patient's lungs or chest area. However, it is important to consider the patient's clinical history and symptoms, as well as consult a healthcare professional for a thorough evaluation and proper diagnosis of any underlying conditions that may not be visible on the X-ray.
```

---

## sample_07 | test_idx=357 | GT=0 | prompt_v2_strict_uncertainty

![sample_07](./reports/task2/images/sample_07_idx357_gt0.png)

**Prompt used:**

```text
You are a radiologist. Provide a concise chest X-ray report with:
FINDINGS: 2-4 bullet points.
IMPRESSION: 1-2 lines.
If you are uncertain due to low resolution, explicitly state uncertainty.
Do NOT hallucinate devices or lines.
Comment specifically on: consolidation/infiltrate, pleural effusion, pneumothorax, cardiac size.
```

**Generated report:**

```text
[INST] 
You are a radiologist. Provide a concise chest X-ray report with:
FINDINGS: 2-4 bullet points.
IMPRESSION: 1-2 lines.
If you are uncertain due to low resolution, explicitly state uncertainty.
Do NOT hallucinate devices or lines.
Comment specifically on: consolidation/infiltrate, pleural effusion, pneumothorax, cardiac size.
 [/INST] FINDINGS:
1. Consolidation/infiltrate in the right lower lobe.
2. Pleural effusion in the right lower lobe.
3. Pneumothorax in the right lower lobe.
4. Cardiac size is normal.

IMPRESSION:
These findings suggest that there are abnormalities in the right lower lobe of the lung, including consolidation/infiltrate, pleural effusion, and pneumothorax. It is important to consider the patient's clinical history and symptoms, as well as consult a healthcare professional for a thorough evaluation and proper diagnosis of the underlying cause of these findings.
```

---

## sample_08 | test_idx=392 | GT=0 | prompt_v1_simple

![sample_08](./reports/task2/images/sample_08_idx392_gt0.png)

**Prompt used:**

```text
Write a short chest X-ray report with sections FINDINGS and IMPRESSION.
Be clinically grounded.
```

**Generated report:**

```text
[INST] 
Write a short chest X-ray report with sections FINDINGS and IMPRESSION.
Be clinically grounded.
 [/INST] FINDINGS: The chest X-ray appears to be unremarkable, which means that there are no significant abnormalities or findings detected in the image.

IMPRESSION: Based on the unremarkable chest X-ray, it suggests that the patient's lungs and surrounding structures appear normal, without any visible signs of disease or injury. However, it is important to consider the patient's clinical history and symptoms, as well as consult a healthcare professional for a thorough evaluation and proper diagnosis of any underlying conditions that may not be visible on the X-ray.
```

---

## sample_08 | test_idx=392 | GT=0 | prompt_v2_strict_uncertainty

![sample_08](./reports/task2/images/sample_08_idx392_gt0.png)

**Prompt used:**

```text
You are a radiologist. Provide a concise chest X-ray report with:
FINDINGS: 2-4 bullet points.
IMPRESSION: 1-2 lines.
If you are uncertain due to low resolution, explicitly state uncertainty.
Do NOT hallucinate devices or lines.
Comment specifically on: consolidation/infiltrate, pleural effusion, pneumothorax, cardiac size.
```

**Generated report:**

```text
[INST] 
You are a radiologist. Provide a concise chest X-ray report with:
FINDINGS: 2-4 bullet points.
IMPRESSION: 1-2 lines.
If you are uncertain due to low resolution, explicitly state uncertainty.
Do NOT hallucinate devices or lines.
Comment specifically on: consolidation/infiltrate, pleural effusion, pneumothorax, cardiac size.
 [/INST] FINDINGS:
1. Consolidation/infiltrate in the right lower lobe.
2. Pleural effusion in the right lower lobe.
3. Pneumothorax in the right lower lobe.
4. Cardiac size is normal.

IMPRESSION:
The chest X-ray shows consolidation/infiltrate, pleural effusion, and pneumothorax in the right lower lobe. These findings suggest that there may be an underlying issue, such as an infection, inflammation, or other lung abnormality. Further evaluation and clinical correlation are needed to determine the cause and appropriate treatment.
```

---

## sample_09 | test_idx=164 | GT=0 | prompt_v1_simple

![sample_09](./reports/task2/images/sample_09_idx164_gt0.png)

**Prompt used:**

```text
Write a short chest X-ray report with sections FINDINGS and IMPRESSION.
Be clinically grounded.
```

**Generated report:**

```text
[INST] 
Write a short chest X-ray report with sections FINDINGS and IMPRESSION.
Be clinically grounded.
 [/INST] FINDINGS: The chest X-ray appears to be unremarkable, which means that there are no significant abnormalities or findings detected in the image.

IMPRESSION: Based on the unremarkable chest X-ray, it seems that there are no major issues or concerns related to the patient's lungs or chest area. However, it is important to consider the patient's clinical history and symptoms, as well as consult a healthcare professional for a thorough evaluation and proper diagnosis of any underlying conditions that may not be visible on the X-ray.
```

---

## sample_09 | test_idx=164 | GT=0 | prompt_v2_strict_uncertainty

![sample_09](./reports/task2/images/sample_09_idx164_gt0.png)

**Prompt used:**

```text
You are a radiologist. Provide a concise chest X-ray report with:
FINDINGS: 2-4 bullet points.
IMPRESSION: 1-2 lines.
If you are uncertain due to low resolution, explicitly state uncertainty.
Do NOT hallucinate devices or lines.
Comment specifically on: consolidation/infiltrate, pleural effusion, pneumothorax, cardiac size.
```

**Generated report:**

```text
[INST] 
You are a radiologist. Provide a concise chest X-ray report with:
FINDINGS: 2-4 bullet points.
IMPRESSION: 1-2 lines.
If you are uncertain due to low resolution, explicitly state uncertainty.
Do NOT hallucinate devices or lines.
Comment specifically on: consolidation/infiltrate, pleural effusion, pneumothorax, cardiac size.
 [/INST] FINDINGS:
1. Consolidation/infiltrate in the right lower lobe.
2. Pleural effusion.
3. Pneumothorax.
4. Cardiac size.

IMPRESSION:
The chest X-ray shows consolidation/infiltrate in the right lower lobe, pleural effusion, pneumothorax, and an enlarged cardiac silhouette. These findings suggest that there may be an underlying issue affecting the patient's respiratory and cardiovascular systems. Further evaluation and consultation with a healthcare professional are necessary to determine the cause and appropriate treatment for these findings.
```

