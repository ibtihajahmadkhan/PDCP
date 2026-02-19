# Task 2 – CNN vs VLM Comparison (10 Samples)

- CNN checkpoint: `./models/best_resnet18.pt`
- CNN threshold: `0.5`
- Source VLM outputs: `./reports/task2/generated_reports.md`

| test_idx | GT | CNN_prob(pneumonia) | CNN_pred | VLM v1 (IMPRESSION snippet) | VLM v2 (IMPRESSION snippet) |
|---:|---:|---:|---:|---|---|
| 112 | 0 | 0.0038 | 0 | IMPRESSION: Based on the unremarkable chest X-ray, it seems that there are no major issues or concerns related to the pa… | 2. Pleural effusion in the right lower lobe. |
| 124 | 1 | 1.0000 | 1 | IMPRESSION: Based on the unremarkable findings in the chest X-ray, it suggests that the patient's lungs and surrounding … | 2. Pleural effusion in the right lower lobe. |
| 145 | 0 | 0.0000 | 0 | IMPRESSION: Based on the normal chest X-ray, the patient's condition seems to be stable. However, it is important to con… | 2. Pleural effusion in the right lower lobe. |
| 164 | 0 | 0.0000 | 0 | IMPRESSION: Based on the unremarkable chest X-ray, it seems that there are no major issues or concerns related to the pa… | 2. Pleural effusion. |
| 357 | 0 | 0.0001 | 0 | IMPRESSION: Based on the unremarkable chest X-ray, it seems that there are no major issues or concerns related to the pa… | 2. Pleural effusion in the right lower lobe. |
| 392 | 0 | 0.0001 | 0 | IMPRESSION: Based on the unremarkable chest X-ray, it suggests that the patient's lungs and surrounding structures appea… | 2. Pleural effusion in the right lower lobe. |
| 393 | 1 | 0.9895 | 1 | IMPRESSION: Based on the unremarkable chest X-ray, it suggests that the patient's lungs and surrounding structures appea… | 2. Pleural effusion. |
| 495 | 1 | 0.9994 | 1 | IMPRESSION: Based on the unremarkable chest X-ray, it seems that there are no major issues or concerns related to the pa… | 2. Pleural effusion. |
| 555 | 1 | 1.0000 | 1 | IMPRESSION: Based on the unremarkable chest X-ray, it seems that there are no major issues or concerns related to the pa… | 2. Pleural effusion in the right lower lobe. |
| 573 | 1 | 0.9996 | 1 | IMPRESSION: Based on the unremarkable chest X-ray, it seems that there are no major issues or concerns related to the pa… | 2. Pleural effusion. |
