"""数据库内检索的模拟连接测试，不代替实际余弦排序验收。"""

from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from context_lab.embeddings import EmbeddingHit, PostgresEmbeddingStore
from context_lab.embeddings.errors import EmbeddingStorageError


class EmbeddingSearchReadyTests(unittest.TestCase):
    def setUp(self):
        self.store = PostgresEmbeddingStore("postgresql://invalid.example/test", schema="test_search")
        self.conn = MagicMock()
        self.conn.__enter__.return_value = self.conn
        self.conn.__exit__.return_value = False
        patcher = patch.object(self.store, "connect", return_value=self.conn)
        self.connect = patcher.start()
        self.addCleanup(patcher.stop)
        self.profile = (2,)
        self.columns = ["profile_id", "namespace", "source_id", "chunk_id", "text",
                        "content_hash", "metadata", "state", "distance"]
        self.record = {"profile_id": "p", "namespace": "code:one", "source_id": "a.py", "chunk_id": "a",
                       "text": "text", "content_hash": "hash-a", "metadata": {"line": [1]},
                       "state": "ready", "distance": 0.2}
        self.rows = [tuple(self.record[name] for name in self.columns)]
        self.description = [SimpleNamespace(name=name) for name in self.columns]
        self.calls = []
        self.conn.execute.side_effect = self.execute

    def execute(self, query, params):
        text = query.as_string()
        self.calls.append((text, params))
        result = MagicMock()
        if "SELECT dimensions" in text:
            result.fetchone.return_value = self.profile
        else:
            result.description = self.description
            result.fetchall.return_value = self.rows
        return result

    def test_database_order_limit_and_hit_score_without_loading_vectors(self):
        hit = self.store.search_ready("p", "code:one", [1.0, 2.0], limit=3)[0]
        self.assertIsInstance(hit, EmbeddingHit)
        self.assertAlmostEqual(hit.score, 0.8)
        hit.chunk.metadata["line"].append(2)
        self.assertEqual(self.record["metadata"], {"line": [1]})
        query, params = self.calls[1]
        self.assertIn('JOIN "test_search"."vectors" AS v', query)
        self.assertIn("c.profile_id = v.profile_id AND c.content_hash = v.content_hash", query)
        self.assertIn("v.embedding <=> %s::vector AS distance", query)
        self.assertIn("ORDER BY v.embedding <=> %s::vector ASC, c.source_id, c.chunk_id LIMIT %s", query)
        self.assertEqual(params[0].to_list(), [1.0, 2.0])
        self.assertEqual(params[1:3], ("p", "code:one"))
        self.assertEqual(params[3].to_list(), [1.0, 2.0])
        self.assertEqual(params[4], 3)
        self.assertEqual(len(self.calls), 2)

    def test_source_filter_has_correct_parameter_position(self):
        self.store.search_ready("p", "code:one", [1.0, 2.0], source_id="a.py")
        query, params = self.calls[1]
        self.assertIn("AND c.source_id = %s", query)
        self.assertEqual(params[1:4], ("p", "code:one", "a.py"))
        self.assertEqual(params[4].to_list(), [1.0, 2.0])
        self.assertEqual(params[5], 5)

    def test_unknown_or_not_yet_embedded_profile_returns_empty_without_distance_query(self):
        for profile in (None, (None,)):
            self.calls.clear()
            self.profile = profile
            with self.subTest(profile=profile):
                self.assertEqual(self.store.search_ready("p", "code:one", [1.0, 2.0]), [])
                self.assertEqual(len(self.calls), 1)

    def test_no_matching_chunks_returns_empty(self):
        self.rows = []
        self.assertEqual(self.store.search_ready("p", "code:one", [1.0, 2.0]), [])

    def test_bad_dimensions_stop_before_distance_query(self):
        for profile in ((3,), (0,), ("2",)):
            self.calls.clear()
            self.profile = profile
            with self.subTest(profile=profile), self.assertRaises(EmbeddingStorageError):
                self.store.search_ready("p", "code:one", [1.0, 2.0])
            self.assertEqual(len(self.calls), 1)

    def test_invalid_inputs_never_connect(self):
        for args in (("", "n", [1.0]), ("p", None, [1.0]), ("p", "n", [1.0], 0),
                     ("p", "n", [1.0], True), ("p", "n", [1.0], 2, " ")):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.store.search_ready(*args)
        for vector in ([], [0, 0], [True], [float("nan")], [float("inf")], [1e100], [1e-100], "12"):
            with self.subTest(vector_type=type(vector).__name__), self.assertRaises(ValueError):
                self.store.search_ready("p", "n", vector)
        self.connect.assert_not_called()

    def test_corrupt_or_out_of_scope_hits_raise(self):
        for changes in ({"profile_id": "other"}, {"namespace": "other"}, {"source_id": "b.py"},
                        {"state": "pending"}, {"metadata": []}, {"distance": None},
                        {"distance": float("nan")}, {"distance": True}, {"text": " "}):
            record = {**self.record, **changes}
            self.rows = [tuple(record[name] for name in self.columns)]
            with self.subTest(changes=changes), self.assertRaises(EmbeddingStorageError):
                self.store.search_ready("p", "code:one", [1.0, 2.0], source_id="a.py")

    def test_bad_columns_and_short_rows_raise(self):
        self.description = None
        with self.assertRaisesRegex(EmbeddingStorageError, "返回列"):
            self.store.search_ready("p", "code:one", [1.0, 2.0])
        self.description = [SimpleNamespace(name=name) for name in self.columns]
        self.rows = [("p",)]
        with self.assertRaisesRegex(EmbeddingStorageError, "字段数量"):
            self.store.search_ready("p", "code:one", [1.0, 2.0])

    def test_query_error_is_not_empty_result_and_hides_details(self):
        self.conn.execute.side_effect = RuntimeError("fake-test-password")
        with self.assertRaises(EmbeddingStorageError) as error:
            self.store.search_ready("p", "code:one", [1.0, 2.0])
        self.assertNotIn("fake-test-password", str(error.exception))
        self.assertIn("RuntimeError", str(error.exception))
