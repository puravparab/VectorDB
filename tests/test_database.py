import tempfile
import unittest
from pathlib import Path

from vectordb.database import VectorDatabase
from vectordb.filters import all_of, equals, greater_than


class VectorDatabaseTests(unittest.TestCase):
    def test_each_index_backend(self):
        options = {
            "brute_force": {},
            "hnsw": {"m": 4, "ef_construction": 20, "seed": 2},
            "ivf": {"n_lists": 2, "seed": 2},
            "kd_tree": {},
            "lsh": {"tables": 6, "hash_size": 6, "seed": 2},
        }
        for backend, backend_options in options.items():
            with self.subTest(backend=backend):
                database = VectorDatabase(2, index=backend, index_options=backend_options)
                database.add("a", [0, 0], {"group": "left"})
                database.add("b", [10, 10], {"group": "right"})
                self.assertEqual(database.search([0, 0], k=1)[0].label, "a")

    def test_metadata_is_copied_at_boundaries(self):
        database = VectorDatabase(1, index="brute_force")
        source = {"kind": "first"}
        database.add("a", [1], source)
        source["kind"] = "changed"
        returned = database.get_metadata("a")
        returned["kind"] = "also changed"
        self.assertEqual(database.get_metadata("a"), {"kind": "first"})
        database.update_metadata("a", {"score": 2})
        self.assertEqual(database.get_metadata("a")["score"], 2)

    def test_remove_updates_vectors_and_metadata(self):
        database = VectorDatabase(1)
        database.add("a", [1], {"x": 1})
        database.remove("a")
        self.assertNotIn("a", database)
        with self.assertRaises(KeyError):
            database.get_metadata("a")

    def test_rejects_unknown_backend(self):
        with self.assertRaises(ValueError):
            VectorDatabase(2, index="unknown")

    def test_filtered_search(self):
        database = VectorDatabase(1, index="brute_force")
        database.add_many(
            [
                ("cheap", [0], {"kind": "book", "price": 5}),
                ("target", [2], {"kind": "book", "price": 20}),
                ("film", [2.1], {"kind": "film", "price": 25}),
            ]
        )
        where = all_of(equals("kind", "book"), greater_than("price", 10))
        self.assertEqual(database.search([2.1], k=2, where=where)[0].label, "target")

    def test_batch_search_and_atomic_add(self):
        database = VectorDatabase(1, index="brute_force")
        database.add_many([("a", [0], None), ("b", [10], None)])
        results = database.search_many([[1], [9]], k=1)
        self.assertEqual([batch[0].label for batch in results], ["a", "b"])
        with self.assertRaises(ValueError):
            database.add_many([("temporary", [5], None), ("a", [2], None)])
        self.assertNotIn("temporary", database)

    def test_save_and_load(self):
        database = VectorDatabase(
            2,
            index="hnsw",
            index_options={"m": 4, "ef_construction": 20, "seed": 4},
        )
        database.add_many(
            [("a", [0, 0], {"kind": "left"}), ("b", [4, 4], {"kind": "right"})]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "database.json"
            database.save(path)
            loaded = VectorDatabase.load(path)
        self.assertEqual(loaded.index_name, "hnsw")
        self.assertEqual(loaded.get_metadata("b"), {"kind": "right"})
        self.assertEqual(loaded.search([4, 4], k=1)[0].label, "b")


if __name__ == "__main__":
    unittest.main()
