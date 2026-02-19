import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from medmnist import PneumoniaMNIST

# ---- allow imports from project root ----
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from task1_classification.model import ResNet18_Gray  # your model


# ---- paths (adjust only if your names differ) ----
CNN_CKPT_PATH = "./models/best_resnet18.pt"  # your trained checkpoint
OUT_DIR = "./reports/task3"
EMB_PATH = os.path.join(OUT_DIR, "embeddings_test.npy")
LAB_PATH = os.path.join(OUT_DIR, "labels_test.npy")
IDX_PATH = os.path.join(OUT_DIR, "test_indices.npy")

IMG_DIR = os.path.join(OUT_DIR, "images_test")  # optional: save some images for visualization
os.makedirs(IMG_DIR, exist_ok=True)


def load_task1_model_as_embedder(device: str):
    model = ResNet18_Gray(pretrained=False)

    ckpt = torch.load(CNN_CKPT_PATH, map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    model.load_state_dict(state, strict=True)

    print("Replacing model.backbone.fc with Identity()")
    model.backbone.fc = nn.Identity()

    model.to(device)
    model.eval()
    return model



def get_eval_transform(img_size=224):
    # Use same preprocessing used in Task 1 evaluation
    # If your Task 1 normalization differs, match it here.
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229]),
    ])


@torch.inference_mode()
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    assert os.path.exists(CNN_CKPT_PATH), f"Missing checkpoint: {CNN_CKPT_PATH}"

    # dataset
    ds_test = PneumoniaMNIST(root="./data", split="test", download=True)
    tfm = get_eval_transform(img_size=224)

    # model
    model = load_task1_model_as_embedder(device)

    embeddings = []
    labels = []
    indices = []

    for i in range(len(ds_test)):
        img, y = ds_test[i]              # img is PIL, y is np array shape (1,)
        x = tfm(img).unsqueeze(0).to(device)  # [1,1,224,224]

        emb = model(x)  # now should be [1,512]
        emb = emb.detach().float().cpu().numpy().reshape(-1)

        embeddings.append(emb)
        labels.append(int(y[0]))
        indices.append(i)

        if (i + 1) % 100 == 0:
            print(f"processed {i+1}/{len(ds_test)}")

    embeddings = np.stack(embeddings, axis=0).astype(np.float32)
    labels = np.array(labels, dtype=np.int64)
    indices = np.array(indices, dtype=np.int64)

    np.save(EMB_PATH, embeddings)
    np.save(LAB_PATH, labels)
    np.save(IDX_PATH, indices)

    print("\nSaved:")
    print(" ", EMB_PATH, embeddings.shape, embeddings.dtype)
    print(" ", LAB_PATH, labels.shape, labels.dtype)
    print(" ", IDX_PATH, indices.shape, indices.dtype)


if __name__ == "__main__":
    main()
