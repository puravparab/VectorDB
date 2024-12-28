import unittest

from vectordb.brute_force import BruteForceIndex


class BruteForceIndexTests(unittest.TestCase):
    def test_exact_results(self):
        index = BruteForceIndex(2)
        index.add_many([("a", [0, 0]), ("b", [2, 0]), ("c", [5, 0])])
        results = index.search([1.8, 0], k=2)
        self.assertEqual([label for label, _ in results], ["b", "a"])
        self.assertAlmostEqual(results[0][1], 0.2)
        self.assertAlmostEqual(results[1][1], 1.8)

    def test_cosine_results(self):
        index = BruteForceIndex(2, metric="cosine")
        index.add_many([("x", [1, 0]), ("y", [0, 1])])
        self.assertEqual(index.search([2, 0], k=1)[0][0], "x")

    def test_remove_and_get_vector(self):
        index = BruteForceIndex(1)
        index.add("point", [3])
        self.assertEqual(index.get_vector("point"), (3.0,))
        index.remove("point")
        self.assertEqual(len(index), 0)
        with self.assertRaises(KeyError):
            index.get_vector("point")

    def test_rejects_duplicate_labels(self):
        index = BruteForceIndex(1)
        index.add(1, [1])
        with self.assertRaises(ValueError):
            index.add(1, [2])


if __name__ == "__main__":
    unittest.main()
