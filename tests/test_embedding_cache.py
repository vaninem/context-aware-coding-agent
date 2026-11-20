"""缓存读取边界测试，使用模拟连接，不连接真实 PostgreSQL。"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from pgvector import Vector

from context_lab.embeddings import PostgresEmbeddingStore
from context_lab.embeddings.errors import EmbeddingStorageError


class EmbeddingCacheTests(unittest.TestCase):
    def setUp(self):
        self.store = PostgresEmbeddingStore("postgresql://invalid.example/test", schema="test_cache")
        self.conn = MagicMock()
        self.conn.__enter__.return_value = self.conn
        self.conn.__exit__.return_value = False
        self.connect = patch.object(self.store, "connect", return_value=self.conn).start()
        self.addCleanup(patch.stopall)

    def test_reads_numpy_vector_and_returns_plain_float_list(self):
        self.conn.execute.return_value.fetchone.return_value = (np.array([1, 2], dtype=np.float32),)
        result = self.store.get_cached("profile-a", "hash-a")
        self.assertEqual(result, [1.0, 2.0])
        self.assertTrue(all(type(value) is float for value in result))
        query, params = self.conn.execute.call_args.args
        self.assertIn('SELECT embedding FROM "test_cache"."vectors"', query.as_string())
        self.assertIn("WHERE profile_id = %s AND content_hash = %s", query.as_string())
        self.assertEqual(params, ("profile-a", "hash-a"))
        self.conn.__exit__.assert_called_once_with(None, None, None)

    def test_missing_record_returns_none(self):
        self.conn.execute.return_value.fetchone.return_value = None
        self.assertIsNone(self.store.get_cached("p", "h"))

    def test_reads_native_pgvector_object(self):
        self.conn.execute.return_value.fetchone.return_value = (Vector([1.0, 2.0]),)
        self.assertEqual(self.store.get_cached("p", "h"), [1.0, 2.0])

    def test_query_values_are_parameters_not_sql_identifiers(self):
        self.conn.execute.return_value.fetchone.return_value = None
        value = "x'; DROP TABLE vectors; --"
        self.store.get_cached(value, "h")
        query, params = self.conn.execute.call_args.args
        self.assertNotIn(value, query.as_string())
        self.assertEqual(params, (value, "h"))

    def test_invalid_arguments_never_connect(self):
        for profile, digest in (("", "h"), ("p", " "), (None, "h"), ("p", 1)):
            with self.subTest(profile=profile, digest=digest), self.assertRaises(ValueError):
                self.store.get_cached(profile, digest)
        self.connect.assert_not_called()

    def test_corrupt_records_raise_instead_of_returning_none(self):
        bad_rows = [(), (None,), ([],), ([0, 0],), ([True],), ([float("nan")],),
                    ([float("inf")],), ("123",), (["1"],), ([[1, 2]],),
                    (np.array([True]),), ([10 ** 400],)]
        for row in bad_rows:
            self.conn.execute.return_value.fetchone.return_value = row
            with self.subTest(row_type=type(row).__name__), self.assertRaisesRegex(EmbeddingStorageError, "缓存向量损坏"):
                self.store.get_cached("p", "h")

    def test_execute_and_fetch_failures_are_redacted(self):
        for stage in ("execute", "fetch"):
            self.conn.execute.side_effect = None
            self.conn.execute.return_value.fetchone.side_effect = None
            failure = RuntimeError("fake-test-password")
            if stage == "execute":
                self.conn.execute.side_effect = failure
            else:
                self.conn.execute.return_value.fetchone.side_effect = failure
            with self.subTest(stage=stage), self.assertRaises(EmbeddingStorageError) as error:
                self.store.get_cached("p", "h")
            self.assertNotIn("fake-test-password", str(error.exception))
            self.assertIn("RuntimeError", str(error.exception))

    def test_connection_failure_is_not_a_cache_miss(self):
        self.connect.side_effect = EmbeddingStorageError("test connection failure")
        with self.assertRaisesRegex(EmbeddingStorageError, "test connection failure"):
            self.store.get_cached("p", "h")
