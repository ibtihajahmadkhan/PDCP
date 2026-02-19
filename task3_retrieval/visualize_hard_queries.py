import os
import numpy as np
import faiss
import matplotlib.pyplot as plt
from medmnist import PneumoniaMNIST


EMB_PATH = "./reports/task3/embeddings_test.npy"
LAB_PATH = "./reports/task3/labels_test.npy"
IDX_PATH = "./reports/task3/test_indices.npy"
INDEX_PATH = "./reports/task3/faiss_index_test.index"

OUT_DIR = "./reports/task3"
OUT_PNG = os.path.join(OUT_DIR, "hard_queries_top5.png")


def l2_normalize(x, eps=1e-12):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)


def main(k=5, worst_n_per_class=3):
    os.makedirs(OUT_DIR, exist_ok=True)

    X = np.load(EMB_PATH).astype(np.float32)
    y = np.load(LAB_PATH).astype(np.int64)
    idx = np.load(IDX_PATH).astype(np.int64)

    Xn = l2_normalize(X)
    index = faiss.read_index(INDEX_PATH)

    # Retrieve neighbors for all queries
    _, I = index.search(Xn, k + 1)
    I = I[:, 1:]  # remove self

    # Per-query hit-rate in top-k
    hit_rate = (y[I] == y[:, None]).mean(axis=1)  # [N]

    # Pick worst queries per class
    worst = {}
    for cls in [0, 1]:
        cls_ids = np.where(y == cls)[0]
        cls_sorted = cls_ids[np.argsort(hit_rate[cls_ids])]  # ascending => worst first
        worst[cls] = cls_sorted[:worst_n_per_class]

    # Dataset for images
    ds_test = PneumoniaMNIST(root="./data", split="test", download=True)

    # Figure layout:
    # rows = 2 classes * worst_n_per_class
    # cols = 1 (query) + k neighbors
    rows = 2 * worst_n_per_class
    cols = 1 + k

    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.1 * rows))

    def draw_row(r, q_id, title_prefix):
        q_img, q_lab = ds_test[int(idx[q_id])]
        q_arr = np.array(q_img)

        axes[r, 0].imshow(q_arr, cmap="gray")
        axes[r, 0].set_title(
            f"{title_prefix}\nq_idx={idx[q_id]} lab={int(q_lab[0])}\nHit@{k}={hit_rate[q_id]:.2f}",
            fontsize=9
        )
        axes[r, 0].axis("off")

        for j in range(k):
            n_id = I[q_id, j]
            n_img, n_lab = ds_test[int(idx[n_id])]
            n_arr = np.array(n_img)

            axes[r, j + 1].imshow(n_arr, cmap="gray")
            axes[r, j + 1].set_title(
                f"NN{j+1}\nidx={idx[n_id]}\nlab={int(n_lab[0])}",
                fontsize=9
            )
            axes[r, j + 1].axis("off")

    r = 0
    for q_id in worst[0]:
        draw_row(r, q_id, "Worst NORMAL")
        r += 1

    for q_id in worst[1]:
        draw_row(r, q_id, "Worst PNEUMONIA")
        r += 1

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200)
    plt.close()

    print("Saved:", OUT_PNG)
    print("Worst Normal query ids (embedding idx):", worst[0].tolist())
    print("Worst Pneumonia query ids (embedding idx):", worst[1].tolist())


if __name__ == "__main__":
    main(k=5, worst_n_per_class=3)
