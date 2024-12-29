import random
import unittest

from vectordb.lsh import LSHIndex


class LSHIndexTests(unittest.TestCase):
    def test_finds_same_direction(self):
        index = LSHIndex(3, tables=10, hash_size=8, seed=7)
        index.add_many(
            [("x", [1, 0, 0]), ("near-x", [0.9, 0.1, 0]), ("y", [0, 1, 0])]
        )
        self.assertEqual(index.search([1, 0, 0], k=2)[0][0], "x")
        self.assertEqual({label for label, _ in index.search([1, 0, 0], k=2)}, {"x", "near-x"})

    def test_recall_for_clustered_vectors(self):
        randomizer = random.Random(12)
        points = {
            number: [
                (1.0 if coordinate == number % 5 else 0.0) + randomizer.gauss(0, 0.08)
                for coordinate in range(5)
            ]
            for number in range(200)
        }
        index = LSHIndex(5, tables=12, hash_size=10, probe_radius=1, seed=12)
        index.add_many(points.items())
        hits = 0
        for label in range(0, 200, 5):
            if label in {result for result, _ in index.search(points[label], k=5)}:
                hits += 1
        self.assertGreaterEqual(hits, 38)

    def test_remove(self):
        index = LSHIndex(2, seed=1)
        index.add("x", [1, 0])
        index.remove("x")
        self.assertEqual(index.search([1, 0]), [])


if __name__ == "__main__":
    unittest.main()
