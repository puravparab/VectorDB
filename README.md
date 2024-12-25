### what is this?

This project is an attempt to build a fast and performant vector database.

### planned features

- [ ] core vector ops
	- [ ] vector storage (simple in-memory first)
	- [ ] euclidean distance
	- [ ] cosine similarity

- [ ] search
	- [ ] k-nearest neighbor
	- [x] approximate nearest neighbor
	- [x] hnsw

- [ ] queries
	- [ ] simple querying
	- [ ] filtered querying
	- [ ] batch querying

- [ ] metadata

- [ ] storage (save/load from disk)

### HNSW quick start

The Python package includes a dependency-free HNSW index with Euclidean and
cosine distance, deletion, batch insertion, and JSON persistence.

```python
from vectordb import HNSWIndex

index = HNSWIndex(dimensions=3, metric="cosine", seed=42)
index.add("first", [1.0, 0.0, 0.0])
index.add("second", [0.8, 0.2, 0.0])

neighbors = index.search([1.0, 0.1, 0.0], k=2)
index.save("vectors.hnsw.json")
```

Install the package for development and run the test suite with:

```shell
python3 -m pip install -e .
python3 -m unittest discover -s tests
```
