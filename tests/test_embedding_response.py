"""响应解析的边界测试，不联网、不依赖数据库。"""

import unittest

from context_lab.embeddings import EmbeddingConfig, HTTPEmbeddingClient
from context_lab.embeddings.errors import EmbeddingRequestError


class EmbeddingResponseTests(unittest.TestCase):
    def test_reorders_vectors_and_converts_to_float(self):
        client = HTTPEmbeddingClient(EmbeddingConfig())
        self.assertEqual(client._parse_response({"data": [
            {"index": 1, "embedding": [0, 2]},
            {"index": 0, "embedding": [1, 0]},
        ]}, 2), [[1.0, 0.0], [0.0, 2.0]])
        self.assertEqual(client.dimensions, 2)

    def test_empty_batch_does_not_set_dimensions(self):
        client = HTTPEmbeddingClient(EmbeddingConfig())
        self.assertEqual(client._parse_response({"data": []}, 0), [])
        self.assertIsNone(client.dimensions)

    def test_rejects_invalid_counts_and_structure(self):
        client = HTTPEmbeddingClient(EmbeddingConfig())
        for count in (-1, True, 1.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                client._parse_response({"data": []}, count)
        for payload in (None, {}, {"data": {}}, {"data": []}, {"data": [None]}):
            with self.subTest(payload=payload), self.assertRaises(EmbeddingRequestError):
                client._parse_response(payload, 1)

    def test_rejects_invalid_or_duplicate_indices(self):
        client = HTTPEmbeddingClient(EmbeddingConfig())
        for index in (-1, 1, True, 0.0, "0", None):
            with self.subTest(index=index), self.assertRaises(EmbeddingRequestError):
                client._parse_response({"data": [{"index": index, "embedding": [1]}]}, 1)
        with self.assertRaises(EmbeddingRequestError):
            client._parse_response({"data": [
                {"index": 0, "embedding": [1]}, {"index": 0, "embedding": [2]},
            ]}, 2)
        self.assertIsNone(client.dimensions)

    def test_rejects_invalid_vectors(self):
        client = HTTPEmbeddingClient(EmbeddingConfig())
        for vector in (None, "bad", [], [0, 0], [True], ["1"], [float("nan")],
                       [float("inf")], [10 ** 400]):
            with self.subTest(vector_type=type(vector).__name__), self.assertRaises(EmbeddingRequestError):
                client._parse_response({"data": [{"index": 0, "embedding": vector}]}, 1)
        self.assertIsNone(client.dimensions)

    def test_rejects_configured_and_cross_batch_dimension_drift(self):
        client = HTTPEmbeddingClient(EmbeddingConfig(expected_dimensions=3))
        with self.assertRaises(EmbeddingRequestError):
            client._parse_response({"data": [{"index": 0, "embedding": [1, 0]}]}, 1)
        client = HTTPEmbeddingClient(EmbeddingConfig())
        client._parse_response({"data": [{"index": 0, "embedding": [1, 0]}]}, 1)
        with self.assertRaises(EmbeddingRequestError):
            client._parse_response({"data": [{"index": 0, "embedding": [1, 0, 0]}]}, 1)
        self.assertEqual(client.dimensions, 2)

    def test_inconsistent_batch_does_not_commit_dimensions(self):
        client = HTTPEmbeddingClient(EmbeddingConfig())
        with self.assertRaises(EmbeddingRequestError):
            client._parse_response({"data": [
                {"index": 0, "embedding": [1, 0]},
                {"index": 1, "embedding": [1, 0, 0]},
            ]}, 2)
        self.assertIsNone(client.dimensions)
