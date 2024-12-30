import unittest

from vectordb.database import VectorDatabase


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


if __name__ == "__main__":
    unittest.main()
