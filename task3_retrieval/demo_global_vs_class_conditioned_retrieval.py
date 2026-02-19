import os
import numpy as np
import faiss
import matplotlib.pyplot as plt
from medmnist import PneumoniaMNIST


OUT_DIR = "./reports/task3"

EMB_PATH = os.path.join(OUT_DIR, "embeddings_test.npy")
LAB_PATH = os.path.join(OUT_DIR, "labels_test.npy")
IDX_PATH = os.path.join(OUT_DIR, "test_indices.npy")

FAISS_ALL_PATH = os.path.join(OUT_DIR, "faiss_index_test.index")
FAISS_NORM_PATH = os.path.join(OUT_DIR, "faiss_index_normal.index")
FAISS_PNEU_PATH = os.path.join(OUT_DIR, "faiss_index_pneumonia.index")


def l2_normalize(x, eps=1e-12):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)


def ensure_exists(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")


def search(index, query_vec, k, use_cosine=True):
    q = query_vec.astype(np.float32).copy()
    if use_cosine:
        faiss.normalize_L2(q)
    D, I = index.search(q, k)
    return D, I


def main(query_emb_idx=1, k=5, use_cosine=True):
    """
    query_emb_idx: index into embeddings_test.npy (0..623)
    k: number of neighbors to show (excluding self)
    """
    os.makedirs(OUT_DIR, exist_ok=True)

    # Check files
    for p in [EMB_PATH, LAB_PATH, IDX_PATH, FAISS_ALL_PATH, FAISS_NORM_PATH, FAISS_PNEU_PATH]:
        ensure_exists(p)

    emb = np.load(EMB_PATH).astype(np.float32)     # (N,512)
    labels = np.load(LAB_PATH).astype(np.int64)    # (N,)
    test_indices = np.load(IDX_PATH).astype(np.int64)  # (N,)

    # Normalize embeddings for cosine retrieval
    emb_n = emb.copy()
    if use_cosine:
        faiss.normalize_L2(emb_n)

    # Load indices
    index_all = faiss.read_index(FAISS_ALL_PATH)
    index_0 = faiss.read_index(FAISS_NORM_PATH)
    index_1 = faiss.read_index(FAISS_PNEU_PATH)

    # Dataset for images
    ds_test = PneumoniaMNIST(root="./data", split="test", download=True)

    # Query
    q_vec = emb_n[query_emb_idx:query_emb_idx + 1]  # (1,512)
    q_label = int(labels[query_emb_idx])
    q_test_idx = int(test_indices[query_emb_idx])
    q_img, q_gt = ds_test[q_test_idx]
    q_gt = int(q_gt[0])

    # ---- Global retrieval (k+1, drop self) ----
    _, I_all = search(index_all, q_vec, k + 1, use_cosine=use_cosine)
    nn_all = I_all[0][1:]  # embeddings indices

    # ---- Class-conditioned retrieval ----
    # Need mapping from class-subset index -> global embedding index
    idx0 = np.where(labels == 0)[0]
    idx1 = np.where(labels == 1)[0]

    if q_label == 0:
        # find its position inside idx0
        q_sub = int(np.where(idx0 == query_emb_idx)[0][0])
        q_vec_sub = emb_n[idx0[q_sub]:idx0[q_sub] + 1]
        _, I_sub = search(index_0, q_vec_sub, k + 1, use_cosine=use_cosine)
        nn_sub_global = idx0[I_sub[0][1:]]
    else:
        q_sub = int(np.where(idx1 == query_emb_idx)[0][0])
        q_vec_sub = emb_n[idx1[q_sub]:idx1[q_sub] + 1]
        _, I_sub = search(index_1, q_vec_sub, k + 1, use_cosine=use_cosine)
        nn_sub_global = idx1[I_sub[0][1:]]

    # ---- Plot (3 rows x (k+1) cols) ----
    # Row 0: Query
    # Row 1: Global neighbors
    # Row 2: Class-conditioned neighbors
    cols = k + 1
    fig, axes = plt.subplots(3, cols, figsize=(2.2 * cols, 6.5))

    def draw(ax, test_idx, lab, title):
        img, gt = ds_test[int(test_idx)]
        gt = int(gt[0])
        ax.imshow(np.array(img), cmap="gray")
        ax.set_title(f"{title}\nidx={int(test_idx)}\nemb_lab={int(lab)}\nGT={gt}", fontsize=9)
        ax.axis("off")

    # Row 0: Query only in first col
    for c in range(cols):
        axes[0, c].axis("off")
    axes[0, 0].imshow(np.array(q_img), cmap="gray")
    axes[0, 0].set_title(f"QUERY\nidx={q_test_idx}\nGT={q_gt}\nemb_idx={query_emb_idx}", fontsize=9)
    axes[0, 0].axis("off")

    # Row 1: Global neighbors
    axes[1, 0].axis("off")
    axes[1, 0].set_title("GLOBAL\n(top-k)", fontsize=10)
    for j, emb_i in enumerate(nn_all, start=1):
        t_idx = int(test_indices[emb_i])
        lab = int(labels[emb_i])
        draw(axes[1, j], t_idx, lab, f"NN{j}")

    # Row 2: Class-conditioned neighbors
    axes[2, 0].axis("off")
    axes[2, 0].set_title("CLASS-CONDITIONED\n(top-k)", fontsize=10)
    for j, emb_i in enumerate(nn_sub_global, start=1):
        t_idx = int(test_indices[emb_i])
        lab = int(labels[emb_i])
        draw(axes[2, j], t_idx, lab, f"NN{j}")

    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, f"global_vs_class_conditioned_idx_{query_emb_idx}.png")
    plt.savefig(out_png, dpi=200)
    plt.close()

    print("Saved:", out_png)
    print("Query: emb_idx =", query_emb_idx, "| test_idx =", q_test_idx, "| GT =", q_gt)
    print("Global neighbor embedding idx:", nn_all.tolist())
    print("Class-conditioned neighbor embedding idx:", nn_sub_global.tolist())


if __name__ == "__main__":
    # Change query_emb_idx to test different queries.
    # Try 1, 9, 67 (worst normal) or 88, 148, 571 (worst pneumonia)
    main(query_emb_idx=88, k=5, use_cosine=True)
