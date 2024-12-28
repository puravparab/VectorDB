import math
import unittest

from vectordb.distance import cosine, distance, euclidean, normalize_vector, squared_euclidean


class DistanceTests(unittest.TestCase):
    def test_euclidean_distance(self):
        self.assertEqual(squared_euclidean([0, 0], [3, 4]), 25)
        self.assertEqual(euclidean([0, 0], [3, 4]), 5)
        self.assertEqual(distance([0, 0], [3, 4], "euclidean"), 5)

    def test_cosine_distance(self):
        self.assertAlmostEqual(cosine([1, 0], [1, 0]), 0)
        self.assertAlmostEqual(cosine([1, 0], [0, 1]), 1)
        self.assertAlmostEqual(cosine([1, 0], [-1, 0]), 2)

    def test_vector_validation(self):
        self.assertEqual(normalize_vector([1, 2], 2, "euclidean"), (1.0, 2.0))
        for vector in ([1], [1, 2, 3], [math.inf, 0], [math.nan, 0]):
            with self.assertRaises(ValueError):
                normalize_vector(vector, 2, "euclidean")
        with self.assertRaises(ValueError):
            normalize_vector([0, 0], 2, "cosine")

    def test_bad_metric(self):
        with self.assertRaises(ValueError):
            distance([0], [0], "manhattan")


if __name__ == "__main__":
    unittest.main()
