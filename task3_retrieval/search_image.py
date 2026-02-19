import os
import numpy as np
import faiss
import matplotlib.pyplot as plt
from medmnist import PneumoniaMNIST


EMB_PATH = "./reports/task3/embeddings_test.npy"
LAB_PATH = "./reports/task3/labels_test.npy"
IDX_PATH = "./reports/task3/test_indices.npy"
INDEX_PATH = "./reports/task3/faiss_index_test.index"


def l2_normalize(x, eps=1e-12):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norm + eps)


def main(query_idx=0, k=5):
    assert os.path.exists(EMB_PATH)
    assert os.path.exists(INDEX_PATH)

    X = np.load(EMB_PATH).astype(np.float32)
    y = np.load(LAB_PATH).astype(np.int64)
    idx = np.load(IDX_PATH).astype(np.int64)

    Xn = l2_normalize(X)

    index = faiss.read_index(INDEX_PATH)

    ds_test = PneumoniaMNIST(root="./data", split="test", download=True)

    q_vec = Xn[query_idx:query_idx+1]
    scores, I = index.search(q_vec, k + 1)

    neighbors = I[0][1:]  # remove self

    fig, axes = plt.subplots(1, k + 1, figsize=(15, 3))

    # Query
    q_img, q_lab = ds_test[int(idx[query_idx])]
    axes[0].imshow(np.array(q_img), cmap="gray")
    axes[0].set_title(f"Query\nidx={idx[query_idx]}\nGT={q_lab[0]}")
    axes[0].axis("off")

    # Neighbors
    for i, n in enumerate(neighbors):
        n_img, n_lab = ds_test[int(idx[n])]
        axes[i + 1].imshow(np.array(n_img), cmap="gray")
        axes[i + 1].set_title(f"NN{i+1}\nidx={idx[n]}\nlab={n_lab[0]}")
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # change query_idx to test different images
    main(query_idx=0, k=5)
