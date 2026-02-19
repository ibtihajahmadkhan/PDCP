import os
import csv
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from torchmetrics.classification import BinaryAUROC

from dataloaders import get_pneumoniamnist_loaders
from model import ResNet18_Gray


def run_epoch(model, loader, optimizer, device, train: bool):
    if train:
        model.train()
    else:
        model.eval()

    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    auroc = BinaryAUROC().to(device)

    pbar = tqdm(loader, desc="train" if train else "val", leave=False)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float().view(-1)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            logits = model(x).view(-1)
            loss = criterion(logits, y)

            if train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * x.size(0)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()
        total_correct += (preds == y.long()).sum().item()
        total_n += x.size(0)

        auroc.update(probs, y.long())
        pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / total_n
    acc = total_correct / total_n
    auc = auroc.compute().item()
    return avg_loss, acc, auc


def plot_curve(x, y1, y2, title, ylabel, out_png):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(x, y1, label="train")
    plt.plot(x, y2, label="val")
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # With 224x224, batch_size 64 is a good start on laptop GPUs
    train_loader, val_loader, test_loader = get_pneumoniamnist_loaders(
        data_root="./data",
        batch_size=64,
        num_workers=2,
        img_size=224
    )

    model = ResNet18_Gray(pretrained=True).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    os.makedirs("./models", exist_ok=True)
    os.makedirs("./reports/task1", exist_ok=True)

    # ---- Early stopping params ----
    max_epochs = 30
    patience = 5          # stop if no val_auc improvement for 5 epochs
    min_delta = 1e-4      # required improvement
    best_val_auc = -1.0
    bad_epochs = 0

    ckpt_path = "./models/best_resnet18.pt"
    history = []  # epoch, tr_loss, tr_acc, tr_auc, va_loss, va_acc, va_auc

    for epoch in range(1, max_epochs + 1):
        tr_loss, tr_acc, tr_auc = run_epoch(model, train_loader, optimizer, device, train=True)
        va_loss, va_acc, va_auc = run_epoch(model, val_loader, optimizer, device, train=False)

        history.append([epoch, tr_loss, tr_acc, tr_auc, va_loss, va_acc, va_auc])

        print(f"Epoch {epoch:02d} | "
              f"train: loss={tr_loss:.4f} acc={tr_acc:.4f} auc={tr_auc:.4f} | "
              f"val: loss={va_loss:.4f} acc={va_acc:.4f} auc={va_auc:.4f}")

        # Early stopping on val AUC
        if va_auc > best_val_auc + min_delta:
            best_val_auc = va_auc
            bad_epochs = 0
            torch.save({"model_state": model.state_dict(), "val_auc": va_auc}, ckpt_path)
            print(f"  saved best -> val_auc={va_auc:.4f}")
        else:
            bad_epochs += 1
            print(f"  no improvement (bad_epochs={bad_epochs}/{patience})")

        if bad_epochs >= patience:
            print(f"\nEarly stopping triggered. Best val_auc={best_val_auc:.4f}")
            break

    # Save history CSV
    csv_path = "./reports/task1/history.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_acc", "train_auc", "val_loss", "val_acc", "val_auc"])
        w.writerows(history)
    print("Saved:", csv_path)

    # Plot curves
    epochs = [h[0] for h in history]
    tr_loss = [h[1] for h in history]
    tr_acc  = [h[2] for h in history]
    tr_auc  = [h[3] for h in history]
    va_loss = [h[4] for h in history]
    va_acc  = [h[5] for h in history]
    va_auc  = [h[6] for h in history]

    plot_curve(epochs, tr_loss, va_loss, "Train vs Val Loss", "loss", "./reports/task1/train_val_loss.png")
    plot_curve(epochs, tr_acc,  va_acc,  "Train vs Val Accuracy", "accuracy", "./reports/task1/train_val_acc.png")
    plot_curve(epochs, tr_auc,  va_auc,  "Train vs Val AUC", "auc", "./reports/task1/train_val_auc.png")
    print("Saved plots in ./reports/task1/")

    # Final test using best checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    te_loss, te_acc, te_auc = run_epoch(model, test_loader, optimizer=None, device=device, train=False)
    print(f"\nTEST | loss={te_loss:.4f} acc={te_acc:.4f} auc={te_auc:.4f}")
    print("Best checkpoint:", ckpt_path, "val_auc:", ckpt["val_auc"])


if __name__ == "__main__":
    main()
