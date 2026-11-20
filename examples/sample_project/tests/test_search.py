import unittest

from src.search import search_documents


class SearchTests(unittest.TestCase):
    def test_plain_query(self):
        self.assertEqual(search_documents("agent", ["Agent tutorial", "Python"]), ["Agent tutorial"])

    def test_query_with_surrounding_spaces(self):
        self.assertEqual(search_documents(" agent ", ["Agent", "Python"]), ["Agent"])
