import math
import random
import tempfile
import unittest
from pathlib import Path

from vectordb import HNSWIndex


class HNSWIndexTests(unittest.TestCase):
    def test_empty_index(self):
        index = HNSWIndex(2)
        self.assertEqual(index.search([0, 0], k=3), [])

    def test_finds_nearest_vectors_and_reports_euclidean_distance(self):
        index = HNSWIndex(2, m=4, ef_construction=30, seed=7)
        index.add_many(
            [
                ("origin", [0, 0]),
                ("right", [2, 0]),
                ("up", [0, 3]),
                ("far", [10, 10]),
            ]
        )
        results = index.search([1.8, 0.1], k=3, ef=20)
        self.assertEqual([label for label, _ in results], ["right", "origin", "up"])
        self.assertAlmostEqual(results[0][1], math.sqrt(0.05))

    def test_cosine_distance(self):
        index = HNSWIndex(3, metric="cosine", m=4, ef_construction=20, seed=2)
        index.add_many([("x", [1, 0, 0]), ("same", [2, 0, 0]), ("y", [0, 1, 0])])
        results = index.search([1, 0, 0], k=3, ef=10)
        self.assertEqual({label for label, distance in results if abs(distance) < 1e-12}, {"x", "same"})
        self.assertAlmostEqual(dict(results)["y"], 1.0)

    def test_duplicate_bad_dimensions_and_invalid_numbers(self):
        index = HNSWIndex(2)
        index.add("a", [1, 2])
        with self.assertRaises(ValueError):
            index.add("a", [3, 4])
        with self.assertRaises(ValueError):
            index.add("short", [1])
        with self.assertRaises(ValueError):
            index.search([float("nan"), 0])

    def test_cosine_rejects_zero_vectors(self):
        index = HNSWIndex(2, metric="cosine")
        with self.assertRaises(ValueError):
            index.add("zero", [0, 0])
        with self.assertRaises(ValueError):
            index.search([0, 0])

    def test_equal_distances_support_mixed_label_types(self):
        index = HNSWIndex(1, m=4, ef_construction=20, seed=5)
        index.add_many([("left", [-1]), (2, [1])])
        self.assertEqual({label for label, _ in index.search([0], k=2)}, {"left", 2})

    def test_remove(self):
        index = HNSWIndex(2, m=4, ef_construction=20, seed=11)
        index.add_many((number, [number, 0]) for number in range(10))
        index.remove(5)
        self.assertNotIn(5, index)
        self.assertNotIn(5, [label for label, _ in index.search([5, 0], k=9, ef=20)])
        with self.assertRaises(KeyError):
            index.remove(5)

    def test_removing_nodes_keeps_remaining_graph_searchable(self):
        index = HNSWIndex(1, m=2, ef_construction=20, seed=11)
        index.add_many((number, [number]) for number in range(30))
        for number in range(10, 20):
            index.remove(number)
        self.assertEqual(index.search([29], k=1, ef=30)[0][0], 29)
        self.assertEqual(index.search([0], k=1, ef=30)[0][0], 0)

    def test_save_and_load_preserves_index_and_future_insertions(self):
        index = HNSWIndex(2, m=4, ef_construction=20, seed=19)
        index.add_many((str(number), [number, number % 3]) for number in range(20))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.json"
            index.save(path)
            loaded = HNSWIndex.load(path)
        self.assertEqual(index.search([7.2, 1], k=8, ef=20), loaded.search([7.2, 1], k=8, ef=20))
        index.add("next", [21, 0])
        loaded.add("next", [21, 0])
        self.assertEqual(index._levels["next"], loaded._levels["next"])

    def test_high_recall_against_exact_search(self):
        randomizer = random.Random(123)
        points = {
            number: [randomizer.uniform(-10, 10) for _ in range(4)] for number in range(300)
        }
        index = HNSWIndex(4, m=12, ef_construction=100, seed=123)
        index.add_many(points.items())

        matched = 0
        expected_count = 0
        for _ in range(20):
            query = [randomizer.uniform(-10, 10) for _ in range(4)]
            exact = sorted(
                points,
                key=lambda label: sum((a - b) ** 2 for a, b in zip(query, points[label])),
            )[:5]
            approximate = [label for label, _ in index.search(query, k=5, ef=80)]
            matched += len(set(exact).intersection(approximate))
            expected_count += len(exact)
        self.assertGreaterEqual(matched / expected_count, 0.95)


if __name__ == "__main__":
    unittest.main()
