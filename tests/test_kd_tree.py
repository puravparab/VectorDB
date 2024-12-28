import random
import unittest

from vectordb.brute_force import BruteForceIndex
from vectordb.kd_tree import KDTreeIndex


class KDTreeIndexTests(unittest.TestCase):
    def test_matches_brute_force(self):
        randomizer = random.Random(42)
        points = [(number, [randomizer.random() for _ in range(4)]) for number in range(100)]
        tree = KDTreeIndex(4)
        exact = BruteForceIndex(4)
        tree.add_many(points)
        exact.add_many(points)
        for _ in range(10):
            query = [randomizer.random() for _ in range(4)]
            self.assertEqual(
                [label for label, _ in tree.search(query, k=7)],
                [label for label, _ in exact.search(query, k=7)],
            )

    def test_rebuilds_after_add_and_remove(self):
        tree = KDTreeIndex(2)
        tree.add_many([("left", [-2, 0]), ("right", [2, 0])])
        self.assertEqual(tree.search([1, 0], k=1)[0][0], "right")
        tree.remove("right")
        tree.add("center", [0, 0])
        self.assertEqual(tree.search([1, 0], k=1)[0][0], "center")

    def test_empty_tree(self):
        self.assertEqual(KDTreeIndex(3).search([0, 0, 0]), [])


if __name__ == "__main__":
    unittest.main()
