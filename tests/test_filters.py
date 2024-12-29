import unittest

from vectordb.filters import all_of, any_of, contains, equals, greater_than, less_than, negate


class FilterTests(unittest.TestCase):
    def setUp(self):
        self.metadata = {
            "category": "book",
            "price": 18,
            "tags": ["python", "search"],
            "author": {"country": "US"},
        }

    def test_field_comparisons(self):
        self.assertTrue(equals("category", "book")(self.metadata))
        self.assertTrue(equals("author.country", "US")(self.metadata))
        self.assertTrue(greater_than("price", 10)(self.metadata))
        self.assertTrue(less_than("price", 20)(self.metadata))
        self.assertTrue(contains("tags", "python")(self.metadata))

    def test_missing_and_mismatched_fields_do_not_match(self):
        self.assertFalse(equals("missing", None)(self.metadata))
        self.assertFalse(greater_than("category", 3)(self.metadata))
        self.assertFalse(contains("price", 1)(self.metadata))

    def test_boolean_composition(self):
        book = equals("category", "book")
        cheap = less_than("price", 20)
        self.assertTrue(all_of(book, cheap)(self.metadata))
        self.assertTrue(any_of(equals("category", "film"), book)(self.metadata))
        self.assertTrue(negate(equals("category", "film"))(self.metadata))


if __name__ == "__main__":
    unittest.main()
