# Task 1 – PneumoniaMNIST Classification Report

---

## 1. Objective

The objective of this task was to build a binary classifier to distinguish between:

- **Normal (0)**
- **Pneumonia (1)**

using the PneumoniaMNIST dataset. The system was required to include proper training, evaluation, visualization, and failure analysis.

---

## 2. Model Architecture

### Backbone

- **ResNet18 (ImageNet pretrained)**
- Modified first convolution layer to accept **1-channel grayscale input**
- Final fully-connected layer replaced with a single logit output

### Input Processing

- Original image resolution: **28×28 grayscale**
- Resized to: **224×224**
- Normalization: ImageNet-style single-channel normalization

---

## 3. Training Strategy

### Data Augmentation (Train Only)

To improve generalization and reduce overfitting:

- Random rotation: ±7°
- Random translation: up to 5%
- Resize to 224×224
- Normalization

Validation and test sets used deterministic preprocessing (no augmentation).

---

### Optimization

- **Optimizer:** AdamW  
- **Learning rate:** 1e-4  
- **Weight decay:** 1e-4  
- **Loss function:** BCEWithLogitsLoss  

---

### Early Stopping

Training used **early stopping based on validation AUC**:

- Maximum epochs: 30  
- Patience: 5 epochs  
- Metric: Validation AUC  
- Best model checkpoint saved automatically  

This prevented overfitting and improved test generalization.

---

## 4. Threshold Selection

Instead of using the default threshold (0.5), the decision threshold was tuned using:

- **Balanced Accuracy on the validation set**

Balanced accuracy was selected to avoid bias toward the dominant class and to maintain strong recall without excessive false positives.

---

## 5. Final Test Results

### Confusion Matrix

|               | Pred Normal | Pred Pneumonia |
|---------------|------------|----------------|
| True Normal   | **189**    | 45             |
| True Pneumonia| 0          | **390**        |

---

### Performance Metrics

- **Accuracy:** ~0.927  
- **Recall (Sensitivity):** 1.000  
- **Precision:** ~0.896  
- **Specificity:** ~0.808  
- **Balanced Accuracy:** ~0.904  
- **AUC:** ~0.984  

---

## 6. Interpretation

### Strengths

- Perfect sensitivity (no false negatives)
- Strong ranking ability (AUC ≈ 0.98)
- Improved generalization via augmentation
- Controlled false positives compared to earlier models
- Stable validation/test alignment after early stopping

### Observed Tradeoff

Reducing false negatives increases false positives.  
Balanced threshold selection provided a practical compromise between sensitivity and specificity.

---

## 7. Failure Case Analysis

Misclassified normal cases often:

- Contained subtle opacity patterns
- Exhibited contrast artifacts
- Appeared visually similar to mild pneumonia

This indicates the model is highly sensitive to lung density variations.

---

## 8. Limitations

- Dataset is relatively small
- Images are low resolution (28×28)
- No external validation dataset used
- No calibration analysis performed

---

## 9. Potential Improvements

- Add test-time augmentation (TTA)
- Calibrate probabilities (Platt scaling)
- Use a larger backbone (e.g., ResNet34)
- Use focal loss to further reduce false positives
- Add ensemble averaging

---

## 10. Final Remarks

The final system demonstrates:

- Effective transfer learning
- Robust generalization
- Proper regularization
- Clinically safe operating point
- Strong evaluation discipline

The pipeline is modular, reproducible, and production-ready.
