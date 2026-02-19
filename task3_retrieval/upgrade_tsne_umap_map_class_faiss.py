import os
import numpy as np
import matplotlib.pyplot as plt

import faiss

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# UMAP is optional; install if missing:
# pip install umap-learn
try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False


# -----------------------------
# Paths
# -----------------------------
OUT_DIR = "./reports/task3"

EMB_PATH = os.path.join(OUT_DIR, "embeddings_test.npy")   # (N,512) float32
LAB_PATH = os.path.join(OUT_DIR, "labels_test.npy")       # (N,) int64
IDX_PATH = os.path.join(OUT_DIR, "test_indices.npy")      # (N,) int64

FAISS_ALL_PATH = os.path.join(OUT_DIR, "faiss_index_test.index")
FAISS_NORM_PATH = os.path.join(OUT_DIR, "faiss_index_normal.index")
FAISS_PNEU_PATH = os.path.join(OUT_DIR, "faiss_index_pneumonia.index")

TSNE_PNG = os.path.join(OUT_DIR, "tsne_embeddings.png")
UMAP_PNG = os.path.join(OUT_DIR, "umap_embeddings.png")

METRICS_TXT = os.path.join(OUT_DIR, "metrics_map_and_precision.txt")


# -----------------------------
# Utility: build FAISS indices
# -----------------------------
def build_faiss_index(emb: np.ndarray, use_cosine: bool = True):
    """
    Build a FAISS index.
    If use_cosine=True, we L2-normalize embeddings and use inner-product (IP),
    which equals cosine similarity for normalized vectors.
    """
    x = emb.astype(np.float32).copy()
    if use_cosine:
        faiss.normalize_L2(x)
        index = faiss.IndexFlatIP(x.shape[1])
    else:
        index = faiss.IndexFlatL2(x.shape[1])
    index.add(x)
    return index


def search_index(index, query_emb, k: int, use_cosine: bool = True):
    q = query_emb.astype(np.float32).copy()
    if use_cosine:
        faiss.normalize_L2(q)
    D, I = index.search(q, k)
    return D, I


# -----------------------------
# Metrics: Precision@k, AP, mAP
# -----------------------------
def precision_at_k(retrieved_labels: np.ndarray, true_label: int, k: int):
    return float(np.mean(retrieved_labels[:k] == true_label))


def average_precision(retrieved_labels: np.ndarray, true_label: int):
    """
    AP for a single query over an ordered list of retrieved labels.
    Here, 'relevant' means same class as query label.
    """
    rel = (retrieved_labels == true_label).astype(np.int32)
    total_rel = rel.sum()
    if total_rel == 0:
        return 0.0

    hits = 0
    ap = 0.0
    for i, r in enumerate(rel, start=1):
        if r == 1:
            hits += 1
            ap += hits / i
    return ap / total_rel


def eval_retrieval_metrics(index, emb, labels, ks=(1, 5, 10), use_cosine: bool = True):
    """
    Evaluate Precision@k + mAP using leave-one-out retrieval:
    we retrieve (k+1) and drop the first item (self-match).
    """
    N = emb.shape[0]
    max_k = max(ks)

    # search all queries at once
    D, I = search_index(index, emb, k=max_k + 1, use_cosine=use_cosine)

    p_at_k = {k: [] for k in ks}
    APs = []
    APs_by_class = {0: [], 1: []}

    for q in range(N):
        true_lab = int(labels[q])

        # drop self-match (first)
        retrieved = I[q][1:]  # length = max_k
        retrieved_labels = labels[retrieved]

        # precision@k
        for k in ks:
            p_at_k[k].append(precision_at_k(retrieved_labels, true_lab, k))

        # AP over full retrieved list (max_k)
        ap = average_precision(retrieved_labels, true_lab)
        APs.append(ap)
        APs_by_class[true_lab].append(ap)

    results = {
        "precision": {k: float(np.mean(p_at_k[k])) for k in ks},
        "mAP": float(np.mean(APs)),
        "mAP_by_class": {
            0: float(np.mean(APs_by_class[0])) if len(APs_by_class[0]) else 0.0,
            1: float(np.mean(APs_by_class[1])) if len(APs_by_class[1]) else 0.0,
        }
    }
    return results


# -----------------------------
# Visualization: t-SNE / UMAP
# -----------------------------
def plot_2d(points_2d, labels, out_png, title):
    plt.figure(figsize=(6, 6))
    for c, name in [(0, "Normal"), (1, "Pneumonia")]:
        mask = labels == c
        plt.scatter(points_2d[mask, 0], points_2d[mask, 1], s=10, alpha=0.7, label=name)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def run_tsne(emb, labels, out_png):
    # Standardize helps t-SNE/UMAP stability
    x = StandardScaler().fit_transform(emb)
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=42
    )
    x2 = tsne.fit_transform(x)
    plot_2d(x2, labels, out_png, "t-SNE of CNN Embeddings (Test Set)")


def run_umap(emb, labels, out_png):
    if not HAS_UMAP:
        print("UMAP not available. Install with: pip install umap-learn")
        return False

    x = StandardScaler().fit_transform(emb)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=25,
        min_dist=0.1,
        metric="euclidean",
        random_state=42
    )
    x2 = reducer.fit_transform(x)
    plot_2d(x2, labels, out_png, "UMAP of CNN Embeddings (Test Set)")
    return True


# -----------------------------
# Class-conditioned retrieval demo
# -----------------------------
def build_class_conditioned_indices(emb, labels, use_cosine: bool = True):
    emb0 = emb[labels == 0]
    emb1 = emb[labels == 1]

    idx0 = np.where(labels == 0)[0]
    idx1 = np.where(labels == 1)[0]

    index0 = build_faiss_index(emb0, use_cosine=use_cosine)
    index1 = build_faiss_index(emb1, use_cosine=use_cosine)

    return (index0, idx0), (index1, idx1)


def eval_class_conditioned(emb, labels, ks=(1, 5, 10), use_cosine: bool = True):
    """
    For each query, search only within its own class index.
    This measures within-class nearest neighbor consistency.
    """
    (index0, map0), (index1, map1) = build_class_conditioned_indices(emb, labels, use_cosine=use_cosine)

    p_at_k = {k: [] for k in ks}
    APs = []
    APs_by_class = {0: [], 1: []}

    max_k = max(ks)

    for q in range(len(emb)):
        true_lab = int(labels[q])
        q_emb = emb[q:q+1]

        if true_lab == 0:
            D, I = search_index(index0, q_emb, k=max_k + 1, use_cosine=use_cosine)
            retrieved_global = map0[I[0][1:]]  # drop self within class
        else:
            D, I = search_index(index1, q_emb, k=max_k + 1, use_cosine=use_cosine)
            retrieved_global = map1[I[0][1:]]

        retrieved_labels = labels[retrieved_global]

        for k in ks:
            p_at_k[k].append(precision_at_k(retrieved_labels, true_lab, k))

        ap = average_precision(retrieved_labels, true_lab)
        APs.append(ap)
        APs_by_class[true_lab].append(ap)

    results = {
        "precision": {k: float(np.mean(p_at_k[k])) for k in ks},
        "mAP": float(np.mean(APs)),
        "mAP_by_class": {
            0: float(np.mean(APs_by_class[0])) if len(APs_by_class[0]) else 0.0,
            1: float(np.mean(APs_by_class[1])) if len(APs_by_class[1]) else 0.0,
        }
    }
    return results


# -----------------------------
# Main
# -----------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    assert os.path.exists(EMB_PATH), f"Missing: {EMB_PATH}"
    assert os.path.exists(LAB_PATH), f"Missing: {LAB_PATH}"
    assert os.path.exists(IDX_PATH), f"Missing: {IDX_PATH}"

    emb = np.load(EMB_PATH).astype(np.float32)
    labels = np.load(LAB_PATH).astype(np.int64)
    test_indices = np.load(IDX_PATH).astype(np.int64)

    print("Loaded:", emb.shape, labels.shape)

    # 1) Build global FAISS index
    use_cosine = True  # cosine is usually best for deep embeddings
    index_all = build_faiss_index(emb, use_cosine=use_cosine)
    faiss.write_index(index_all, FAISS_ALL_PATH)
    print("Saved global index:", FAISS_ALL_PATH)

    # 2) t-SNE + UMAP
    print("Running t-SNE...")
    run_tsne(emb, labels, TSNE_PNG)
    print("Saved:", TSNE_PNG)

    print("Running UMAP...")
    ok_umap = run_umap(emb, labels, UMAP_PNG)
    if ok_umap:
        print("Saved:", UMAP_PNG)

    # 3) mAP + Precision@k (global)
    ks = (1, 5, 10)
    global_metrics = eval_retrieval_metrics(index_all, emb, labels, ks=ks, use_cosine=use_cosine)

    # 4) class-conditioned FAISS indices + evaluation
    # Save indices for later use (optional)
    emb0 = emb[labels == 0]
    emb1 = emb[labels == 1]
    index0 = build_faiss_index(emb0, use_cosine=use_cosine)
    index1 = build_faiss_index(emb1, use_cosine=use_cosine)
    faiss.write_index(index0, FAISS_NORM_PATH)
    faiss.write_index(index1, FAISS_PNEU_PATH)
    print("Saved class indices:")
    print(" ", FAISS_NORM_PATH)
    print(" ", FAISS_PNEU_PATH)

    class_cond_metrics = eval_class_conditioned(emb, labels, ks=ks, use_cosine=use_cosine)

    # 5) Save metrics report
    with open(METRICS_TXT, "w", encoding="utf-8") as f:
        f.write("Task 3 – Retrieval Metrics Upgrade\n")
        f.write("=================================\n\n")
        f.write(f"Embeddings: {EMB_PATH}  shape={emb.shape}\n")
        f.write(f"Labels    : {LAB_PATH}  shape={labels.shape}\n")
        f.write(f"Index(all): {FAISS_ALL_PATH}\n")
        f.write(f"Index(0)  : {FAISS_NORM_PATH}\n")
        f.write(f"Index(1)  : {FAISS_PNEU_PATH}\n")
        f.write(f"Similarity: {'cosine (IP on L2-normalized)' if use_cosine else 'L2'}\n\n")

        f.write("GLOBAL INDEX (all samples)\n")
        f.write("-------------------------\n")
        for k in ks:
            f.write(f"Precision@{k}: {global_metrics['precision'][k]:.4f}\n")
        f.write(f"mAP: {global_metrics['mAP']:.4f}\n")
        f.write(f"mAP (Normal=0): {global_metrics['mAP_by_class'][0]:.4f}\n")
        f.write(f"mAP (Pneumonia=1): {global_metrics['mAP_by_class'][1]:.4f}\n\n")

        f.write("CLASS-CONDITIONED INDEX (search only within predicted class)\n")
        f.write("-----------------------------------------------------------\n")
        for k in ks:
            f.write(f"Precision@{k}: {class_cond_metrics['precision'][k]:.4f}\n")
        f.write(f"mAP: {class_cond_metrics['mAP']:.4f}\n")
        f.write(f"mAP (Normal=0): {class_cond_metrics['mAP_by_class'][0]:.4f}\n")
        f.write(f"mAP (Pneumonia=1): {class_cond_metrics['mAP_by_class'][1]:.4f}\n\n")

        f.write("VISUALIZATIONS\n")
        f.write("--------------\n")
        f.write(f"t-SNE: {TSNE_PNG}\n")
        if ok_umap:
            f.write(f"UMAP: {UMAP_PNG}\n")
        else:
            f.write("UMAP: not generated (install umap-learn)\n")

    print("Saved metrics:", METRICS_TXT)
    print("\nGLOBAL:", global_metrics)
    print("CLASS-CONDITIONED:", class_cond_metrics)


if __name__ == "__main__":
    main()
