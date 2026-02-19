import os
import numpy as np
import faiss


EMB_PATH = "./reports/task3/embeddings_test.npy"
LAB_PATH = "./reports/task3/labels_test.npy"
INDEX_PATH = "./reports/task3/faiss_index_test.index"


def l2_normalize(x, eps=1e-12):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norm + eps)


def precision_at_k(index, Xn, y, k):
    D, I = index.search(Xn, k + 1)
    I = I[:, 1:]  # remove self-match

    correct = 0
    total = 0

    for i in range(len(Xn)):
        neighbors = I[i]
        correct += np.sum(y[neighbors] == y[i])
        total += k

    return correct / total


def precision_at_k_per_class(index, Xn, y, k):
    D, I = index.search(Xn, k + 1)
    I = I[:, 1:]

    results = {}

    for cls in [0, 1]:
        mask = (y == cls)
        correct = 0
        total = 0

        for i in np.where(mask)[0]:
            neighbors = I[i]
            correct += np.sum(y[neighbors] == y[i])
            total += k

        results[cls] = correct / total

    return results


def main():
    assert os.path.exists(EMB_PATH)
    assert os.path.exists(INDEX_PATH)

    X = np.load(EMB_PATH).astype(np.float32)
    y = np.load(LAB_PATH).astype(np.int64)

    Xn = l2_normalize(X)

    index = faiss.read_index(INDEX_PATH)

    for k in [1, 5, 10]:
        overall = precision_at_k(index, Xn, y, k)
        per_class = precision_at_k_per_class(index, Xn, y, k)

        print(f"\nPrecision@{k}")
        print(f"  Overall     : {overall:.4f}")
        print(f"  Normal (0)  : {per_class[0]:.4f}")
        print(f"  Pneumonia(1): {per_class[1]:.4f}")


if __name__ == "__main__":
    main()
