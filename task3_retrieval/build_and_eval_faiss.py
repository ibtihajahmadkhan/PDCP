import os
import numpy as np
import faiss
import matplotlib.pyplot as plt

from medmnist import PneumoniaMNIST


EMB_PATH = "./reports/task3/embeddings_test.npy"
LAB_PATH = "./reports/task3/labels_test.npy"
IDX_PATH = "./reports/task3/test_indices.npy"

OUT_DIR = "./reports/task3"
INDEX_PATH = os.path.join(OUT_DIR, "faiss_index_test.index")
METRICS_PATH = os.path.join(OUT_DIR, "metrics_precision_at_k.txt")
VIZ_PATH = os.path.join(OUT_DIR, "retrieval_examples.png")


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norm + eps)


def precision_at_k(y_true: np.ndarray, nn_labels: np.ndarray, k: int) -> float:
    """
    y_true: [N] query label
    nn_labels: [N, K] labels of retrieved neighbors (excluding self)
    """
    hits = (nn_labels[:, :k] == y_true[:, None]).mean(axis=1)
    return float(hits.mean())


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    assert os.path.exists(EMB_PATH), f"Missing: {EMB_PATH}"
    assert os.path.exists(LAB_PATH), f"Missing: {LAB_PATH}"
    assert os.path.exists(IDX_PATH), f"Missing: {IDX_PATH}"

    X = np.load(EMB_PATH).astype(np.float32)   # [N, D]
    y = np.load(LAB_PATH).astype(np.int64)     # [N]
    idx = np.load(IDX_PATH).astype(np.int64)   # [N]
    N, D = X.shape

    # Normalize and use cosine similarity via inner product index
    Xn = l2_normalize(X)

    index = faiss.IndexFlatIP(D)  # cosine via normalized dot product
    index.add(Xn)
    faiss.write_index(index, INDEX_PATH)

    # Retrieve K+1 (first neighbor will be itself with score ~1)
    K = 10
    scores, I = index.search(Xn, K + 1)     # I: [N, K+1]
    I = I[:, 1:]                            # drop self

    nn_labels = y[I]                        # [N, K]

    p1 = precision_at_k(y, nn_labels, 1)
    p5 = precision_at_k(y, nn_labels, 5)
    p10 = precision_at_k(y, nn_labels, 10)

    # Save metrics
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        f.write("Task 3 - Retrieval metrics (Test split)\n")
        f.write(f"N={N}, D={D}\n")
        f.write("Similarity: cosine (L2-normalized embeddings + inner product)\n\n")
        f.write(f"Precision@1  = {p1:.4f}\n")
        f.write(f"Precision@5  = {p5:.4f}\n")
        f.write(f"Precision@10 = {p10:.4f}\n")

    print("Saved index:", INDEX_PATH)
    print("Saved metrics:", METRICS_PATH)
    print(f"Precision@1={p1:.4f}  Precision@5={p5:.4f}  Precision@10={p10:.4f}")

    # ---- Visualization: 3 random queries with top-5 neighbors ----
    ds_test = PneumoniaMNIST(root="./data", split="test", download=True)

    rng = np.random.default_rng(0)
    query_ids = rng.choice(N, size=3, replace=False)

    fig, axes = plt.subplots(nrows=3, ncols=6, figsize=(12, 6))
    for r, q in enumerate(query_ids):
        q_img, q_lab = ds_test[int(idx[q])]
        q_arr = np.array(q_img)

        axes[r, 0].imshow(q_arr, cmap="gray")
        axes[r, 0].set_title(f"Q idx={idx[q]}\nGT={q_lab[0]}")
        axes[r, 0].axis("off")

        for c in range(5):
            n = I[q, c]
            n_img, n_lab = ds_test[int(idx[n])]
            n_arr = np.array(n_img)

            axes[r, c + 1].imshow(n_arr, cmap="gray")
            axes[r, c + 1].set_title(f"NN{c+1}\nidx={idx[n]}\nlab={n_lab[0]}")
            axes[r, c + 1].axis("off")

    plt.tight_layout()
    plt.savefig(VIZ_PATH, dpi=200)
    plt.close()
    print("Saved visualization:", VIZ_PATH)


if __name__ == "__main__":
    main()
