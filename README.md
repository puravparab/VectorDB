# VectorDB

A small, dependency-free vector database with an in-memory HNSW index.

```shell
python3 -m pip install -e .
```

```python
from vectordb import HNSWIndex

index = HNSWIndex(dimensions=3, metric="cosine", seed=42)
index.add("first", [1.0, 0.0, 0.0])
index.add("second", [0.8, 0.2, 0.0])

neighbors = index.search([1.0, 0.1, 0.0], k=2)
index.save("vectors.json")
```

## Status

- [x] In-memory vector storage
- [x] Euclidean distance
- [x] Cosine distance
- [x] k-nearest-neighbor search
- [x] Approximate search with HNSW
- [x] Simple queries
- [x] Save and load from disk
- [ ] Filtered queries
- [ ] Batch queries
- [ ] Metadata

Run tests with `python3 -m unittest discover -s tests`.
