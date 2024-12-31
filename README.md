# VectorDB

A small, dependency-free vector database in Python and C++.

Indexes: brute force, KD-tree, LSH, IVF-flat, and HNSW.

```shell
python3 -m pip install -e .
```

```python
from vectordb import VectorDatabase, equals

db = VectorDatabase(3, index="hnsw", index_options={"seed": 42})
db.add("first", [1, 0, 0], {"kind": "example"})
db.add("second", [0.8, 0.2, 0], {"kind": "example"})

results = db.search([1, 0.1, 0], k=2, where=equals("kind", "example"))
db.save("vectors.json")
```

Features include Euclidean and cosine distance, metadata filters, batch queries,
deletion, and JSON persistence.

```shell
python3 -m unittest discover -s tests

cmake -S . -B build
cmake --build build
ctest --test-dir build
```
