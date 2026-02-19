import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

from dataloaders import get_pneumoniamnist_loaders
from model import ResNet18_Gray


def save_confusion_matrix(cm, out_png):
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Normal(0)", "Pneumonia(1)"], rotation=20)
    plt.yticks(tick_marks, ["Normal(0)", "Pneumonia(1)"])

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_roc_curve(y_true, y_prob, out_png):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_failure_cases(images, y_true, y_pred, y_prob, out_png, max_n=25):
    # images: [N,1,28,28] normalized (-1..1). We'll denormalize to [0,1] for saving.
    wrong = np.where(y_true != y_pred)[0]
    if len(wrong) == 0:
        return False

    wrong = wrong[:max_n]
    n = len(wrong)
    grid = int(np.ceil(np.sqrt(n)))

    plt.figure(figsize=(8, 8))
    for k, idx in enumerate(wrong, start=1):
        img = images[idx, 0]
        img = (img * 0.5) + 0.5  # back to [0,1]
        img = np.clip(img, 0, 1)

        plt.subplot(grid, grid, k)
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.title(f"T={y_true[idx]} P={y_pred[idx]}\nProb={y_prob[idx]:.2f}", fontsize=8)

    plt.suptitle("Failure Cases (Misclassified)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    return True


@torch.no_grad()
def infer(model, loader, device):
    model.eval()
    all_imgs = []
    all_y = []
    all_prob = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x).view(-1)
        prob = torch.sigmoid(logits).detach().cpu().numpy()

        all_imgs.append(x.detach().cpu().numpy())
        all_y.append(y.view(-1).cpu().numpy())
        all_prob.append(prob)

    images = np.concatenate(all_imgs, axis=0)
    y_true = np.concatenate(all_y, axis=0).astype(int)
    y_prob = np.concatenate(all_prob, axis=0)
    y_pred = (y_prob >= 0.5).astype(int)
    return images, y_true, y_prob, y_pred


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    train_loader, val_loader, test_loader = get_pneumoniamnist_loaders(
        data_root="./data",
        batch_size=256,
        num_workers=2
    )

    # Load model
    model = model = ResNet18_Gray(pretrained=True).to(device)
    ckpt_path = "./models/best_resnet18.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])

    # Output dirs
    out_dir = "./reports/task1"
    os.makedirs(out_dir, exist_ok=True)

    # First evaluate on validation set to tune threshold
    val_images, val_y_true, val_y_prob, _ = infer(model, val_loader, device)

    from sklearn.metrics import balanced_accuracy_score

    best_thr = 0.5
    best_score = 0

    for thr in np.linspace(0.1, 0.9, 81):
        val_pred = (val_y_prob >= thr).astype(int)
        score = balanced_accuracy_score(val_y_true, val_pred)
        if score > best_score:
            best_score = score
            best_thr = thr

    print(f"Best threshold (Balanced Acc) = {best_thr:.3f}, Score = {best_score:.4f}")

    # Now evaluate TEST using this threshold
    images, y_true, y_prob, _ = infer(model, test_loader, device)
    y_pred = (y_prob >= best_thr).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)

    cm = confusion_matrix(y_true, y_pred)

    # Save plots
    cm_png = os.path.join(out_dir, "confusion_matrix_test.png")
    roc_png = os.path.join(out_dir, "roc_curve_test.png")
    fail_png = os.path.join(out_dir, "failure_cases_test.png")

    save_confusion_matrix(cm, cm_png)
    save_roc_curve(y_true, y_prob, roc_png)
    has_fail = save_failure_cases(images, y_true, y_pred, y_prob, fail_png)

    # Save metrics
    metrics_txt = os.path.join(out_dir, "metrics_test.txt")
    with open(metrics_txt, "w", encoding="utf-8") as f:
        f.write(f"Checkpoint: {ckpt_path}\n")
        f.write(f"Accuracy : {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall   : {rec:.4f}\n")
        f.write(f"F1-score : {f1:.4f}\n")
        f.write(f"AUC      : {auc:.4f}\n")
        f.write("\nConfusion Matrix [ [TN FP] [FN TP] ]:\n")
        f.write(str(cm) + "\n")
        f.write(f"\nFailure cases saved: {has_fail}\n")

    print("Saved:")
    print(" ", metrics_txt)
    print(" ", cm_png)
    print(" ", roc_png)
    if has_fail:
        print(" ", fail_png)

    print("\nTEST metrics:")
    print(f"Accuracy={acc:.4f} Precision={prec:.4f} Recall={rec:.4f} F1={f1:.4f} AUC={auc:.4f}")


if __name__ == "__main__":
    main()
