import random
import unittest

from vectordb.ivf import IVFFlatIndex


class IVFFlatIndexTests(unittest.TestCase):
    def test_clustered_search(self):
        randomizer = random.Random(9)
        points = []
        for cluster, center in enumerate((-20.0, 0.0, 20.0)):
            points.extend(
                (f"{cluster}-{number}", [center + randomizer.gauss(0, 1), randomizer.gauss(0, 1)])
                for number in range(50)
            )
        index = IVFFlatIndex(2, n_lists=6, n_probe=2, seed=9)
        index.add_many(points)
        index.train()
        results = index.search([20, 0], k=10)
        self.assertTrue(all(label.startswith("2-") for label, _ in results))

    def test_auto_train_and_dynamic_updates(self):
        index = IVFFlatIndex(1, n_lists=2, seed=3)
        index.add_many([("left", [-5]), ("right", [5])])
        self.assertEqual(index.search([4], k=1)[0][0], "right")
        index.add("near", [4])
        self.assertEqual(index.search([4], k=1)[0][0], "near")
        index.remove("near")
        self.assertEqual(index.search([4], k=1)[0][0], "right")

    def test_cosine_metric(self):
        index = IVFFlatIndex(2, metric="cosine", n_lists=2, seed=2)
        index.add_many([("x", [1, 0]), ("y", [0, 1]), ("near-x", [1, 0.1])])
        index.train()
        self.assertEqual(index.search([1, 0], k=1)[0][0], "x")


if __name__ == "__main__":
    unittest.main()
