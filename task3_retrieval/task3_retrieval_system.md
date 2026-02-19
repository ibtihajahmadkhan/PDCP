# Task 3 – Semantic Image Retrieval (CNN Embeddings + FAISS)

## 1. Objective

The objective of this task is to build a semantic image retrieval system for PneumoniaMNIST. Given a query chest X-ray image, the system retrieves the most visually and semantically similar images using learned feature embeddings and a vector similarity index.

The system includes:

- CNN-based embedding extraction  
- FAISS vector indexing  
- Image-to-image retrieval  
- Quantitative evaluation using Precision@k and mAP  
- Embedding space visualization (t-SNE / UMAP)  
- Class-conditioned retrieval analysis  

---

## 2. Dataset

- **Dataset:** PneumoniaMNIST (MedMNIST v2)  
- **Split used:** Test split  
- **Total test images:** 624  
- **Labels:**
  - `0 = Normal`
  - `1 = Pneumonia`
- **Resolution:** 28×28 grayscale  

Due to low spatial resolution, fine-grained radiological patterns are limited. Retrieval performance primarily depends on learned global feature representations.

---

## 3. Embedding Model

### 3.1 Model Source

The embedding extractor is derived from the Task 1 classifier:

- Model: `ResNet18_Gray`
- Backbone: ResNet18
- Original classifier head: `Linear(512 → 1)`

### 3.2 Conversion to Embedding Extractor

To convert the classifier into a feature extractor, the final classification layer was removed:

```python
model.backbone.fc = nn.Identity()
```

This modification allows the network to output a **512-dimensional embedding vector** per image instead of a single logit.

### 3.3 Rationale

- Reuses a trained pneumonia-aware feature extractor  
- Produces semantically meaningful embeddings aligned with disease classification  
- Efficient and fully reproducible  
- Serves as a strong baseline for medical image retrieval  

---

## 4. Embedding Extraction

### 4.1 Preprocessing

Each test image undergoes:

- Resize to 224×224  
- Conversion to tensor  
- Normalization:
  - `mean = 0.485`
  - `std = 0.229`

### 4.2 Generated Artifacts

- `reports/task3/embeddings_test.npy` → shape `(624, 512)`  
- `reports/task3/labels_test.npy` → shape `(624,)`  
- `reports/task3/test_indices.npy` → shape `(624,)`  

### 4.3 Script

```
task3_retrieval/extract_embeddings.py
```

---

## 5. FAISS Vector Index

### 5.1 Similarity Metric

Cosine similarity is implemented by:

1. L2-normalizing embeddings  
2. Using FAISS `IndexFlatIP`  

For normalized vectors:

\[
\text{Cosine Similarity} = \text{Inner Product}
\]

### 5.2 Index Construction

- Index type: `faiss.IndexFlatIP`  
- Embedding dimension: 512  
- Number of indexed vectors: 624  

Saved artifact:

```
reports/task3/faiss_index_test.index
```

---

## 6. Retrieval Evaluation

### 6.1 Evaluation Protocol

For each test image used as a query:

1. Retrieve top-(k+1) neighbors  
2. Remove self-retrieval  
3. Compare retrieved labels with query label  

Precision@k is defined as:

\[
\text{Precision@k} =
\frac{1}{N} \sum_{i=1}^{N}
\frac{\#\{\text{top-k retrieved labels} = y_i\}}{k}
\]

---

### 6.2 Global Retrieval Results

The global FAISS index achieves:

- **Precision@1  = 0.9583**  
- **Precision@5  = 0.9474**  
- **Precision@10 = 0.9462**  
- **mAP          = 0.9574**

Class-wise mAP:

- **Normal (0):     0.9232**  
- **Pneumonia (1):  0.9778**

Metrics saved in:

```
reports/task3/metrics_map_and_precision.txt
```

---

## 7. Embedding Space Visualization (t-SNE / UMAP)

To analyze cluster separability, embeddings were projected into 2D using:

- **t-SNE**
- **UMAP**

Generated figures:

- `reports/task3/tsne_embeddings.png`
- `reports/task3/umap_embeddings.png`

Both visualizations show distinguishable clustering between Normal and Pneumonia samples, confirming that the CNN learned semantically meaningful representations.

---

## 8. Class-Conditioned FAISS Indices

Two separate FAISS indices were constructed:

- `faiss_index_normal.index`
- `faiss_index_pneumonia.index`

When restricting retrieval to the query's own class:

- **Precision@1  = 1.0000**
- **Precision@5  = 1.0000**
- **Precision@10 = 1.0000**
- **mAP          = 1.0000**

Class-conditioned retrieval is useful in classifier-guided pipelines but does not replace global evaluation since cross-class confusion is artificially removed.

---

## 9. Global vs Class-Conditioned Retrieval (Qualitative Analysis)

### 9.1 Hard Normal Query (GT = 0)

Query:

- `emb_idx = 1`
- `test_idx = 1`
- `GT = 0`

Global nearest neighbors:

```
[369, 216, 8, 347, 550]
```

All neighbors labeled Pneumonia.

Class-conditioned nearest neighbors:

```
[577, 378, 380, 370, 67]
```

All neighbors labeled Normal.

**Insight:** Global similarity search can cross class boundaries for ambiguous images, while class-conditioned retrieval enforces label consistency.

---

### 9.2 Hard Pneumonia Query (GT = 1)

Query:

- `emb_idx = 88`
- `test_idx = 88`
- `GT = 1`

Global nearest neighbors:

```
[423, 485, 505, 28, 107]
```

All neighbors labeled Normal.

Class-conditioned nearest neighbors:

```
[571, 190, 445, 533, 534]
```

All neighbors labeled Pneumonia.

**Insight:** Subtle pneumonia cases may drift toward the Normal cluster in global embedding space.

---

## 10. Failure Case Analysis

Worst-performing queries based on Hit@5:

### Worst Normal Queries

```
[1, 9, 67]
```

- Hit@5 = 0.00  
- All retrieved neighbors labeled Pneumonia  

### Worst Pneumonia Queries

```
[88, 148, 571]
```

- Hit@5 ranged from 0.00 to 0.40  
- Retrieved neighbors often labeled Normal  

### Systematic Pattern

| Query Type | Failure Mode |
|------------|-------------|
| Normal     | Borderline opacity → clustered with Pneumonia |
| Pneumonia  | Subtle pathology → clustered with Normal |

These patterns reflect low-resolution constraints and texture-based embedding behavior.

---

## 11. Strengths

- End-to-end reproducible retrieval pipeline  
- High Precision@k and mAP  
- Clear embedding cluster separability  
- Efficient FAISS indexing  
- Demonstrated global vs class-conditioned retrieval behavior  

---

## 12. Limitations

- Binary label space  
- 28×28 resolution limits fine-grained discrimination  
- Retrieval evaluated only within test split  
- Image-to-image retrieval only  

---

## 13. Reproducibility

### Install FAISS

```bash
pip install faiss-cpu
```

### Extract Embeddings

```bash
python task3_retrieval/extract_embeddings.py
```

### Build and Evaluate

```bash
python task3_retrieval/build_and_eval_faiss.py
```

### Run Advanced Upgrade

```bash
python task3_retrieval/upgrade_tsne_umap_map_class_faiss.py
```

---

## 14. Conclusion

This task demonstrates that CNN embeddings trained for pneumonia classification can be effectively repurposed for semantic image retrieval.

The system achieves:

- Precision@1 ≈ 0.96  
- mAP ≈ 0.96  

Embedding visualization confirms meaningful class clustering. Failure analysis reveals that retrieval errors are primarily driven by subtle pathology and low image resolution.

Class-conditioned FAISS provides clinically coherent retrieval and is well-suited for hybrid classifier-guided systems.

The resulting framework is scalable, reproducible, and extensible to multi-class or cross-modal medical retrieval applications.
