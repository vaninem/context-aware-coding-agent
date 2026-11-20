"""来源块恢复测试，使用模拟连接，不读取真实数据或访问数据库。"""

from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from context_lab.embeddings import IndexedChunk, PostgresEmbeddingStore
from context_lab.embeddings.errors import EmbeddingStorageError


class EmbeddingSourceTests(unittest.TestCase):
    def setUp(self):
        self.store = PostgresEmbeddingStore("postgresql://invalid.example/test", schema="test_source")
        self.conn = MagicMock()
        self.conn.__enter__.return_value = self.conn
        self.conn.__exit__.return_value = False
        self.cursor = self.conn.execute.return_value
        self.columns = ["profile_id", "namespace", "source_id", "chunk_id", "text", "content_hash", "metadata", "state"]
        self.cursor.description = [SimpleNamespace(name=name) for name in self.columns]
        self.cursor.fetchall.return_value = []
        patcher = patch.object(self.store, "connect", return_value=self.conn)
        self.connect = patcher.start()
        self.addCleanup(patcher.stop)

    def record(self, **changes):
        return {"profile_id": "p", "namespace": "code:one", "source_id": "a.py",
                "chunk_id": "get", "text": "read cache", "content_hash": "hash-a",
                "metadata": {"line": 1}, "state": "ready", **changes}

    def test_rows_restore_objects_and_query_is_scoped_with_explicit_columns(self):
        records = [self.record(), self.record(chunk_id="put", text="write cache", state="pending")]
        self.cursor.fetchall.return_value = [tuple(record[name] for name in self.columns) for record in records]
        results = self.store.list_source("p", "code:one", "a.py")
        self.assertEqual(results, [IndexedChunk(**record) for record in records])
        query, params = self.conn.execute.call_args.args
        text = query.as_string()
        self.assertIn('FROM "test_source"."chunks"', text)
        self.assertIn("WHERE profile_id = %s AND namespace = %s AND source_id = %s", text)
        self.assertIn("ORDER BY chunk_id", text)
        self.assertNotIn("SELECT *", text)
        self.assertEqual(params, ("p", "code:one", "a.py"))
        self.conn.execute.assert_called_once()

    def test_no_records_returns_empty_list(self):
        self.assertEqual(self.store.list_source("p", "code:one", "a.py"), [])

    def test_description_maps_actual_column_order_and_metadata_is_detached(self):
        record = self.record(metadata={"location": {"line": 1}})
        columns = list(reversed(self.columns))
        self.cursor.description = [SimpleNamespace(name=name) for name in columns]
        self.cursor.fetchall.return_value = [tuple(record[name] for name in columns)]
        result = self.store.list_source("p", "code:one", "a.py")[0]
        self.assertEqual(result.chunk_id, "get")
        result.metadata["location"]["line"] = 100
        self.assertEqual(record["metadata"], {"location": {"line": 1}})

    def test_invalid_arguments_do_not_connect(self):
        for args in (("", "n", "s"), ("p", None, "s"), ("p", "n", " ")):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.store.list_source(*args)
        self.connect.assert_not_called()

    def test_corrupt_or_out_of_scope_records_are_not_silently_returned(self):
        changes = [{"profile_id": "other"}, {"namespace": "code:other"}, {"source_id": "b.py"},
                   {"state": "unknown"}, {"metadata": []}, {"chunk_id": ""}, {"text": " "}, {"content_hash": ""}]
        for change in changes:
            record = self.record(**change)
            self.cursor.fetchall.return_value = [tuple(record[name] for name in self.columns)]
            with self.subTest(change=change), self.assertRaises(EmbeddingStorageError):
                self.store.list_source("p", "code:one", "a.py")

    def test_missing_duplicate_columns_and_short_rows_raise_clear_errors(self):
        for names in (None, self.columns[:-1], ["profile_id"] * 8):
            self.cursor.description = None if names is None else [SimpleNamespace(name=name) for name in names]
            with self.subTest(names=names), self.assertRaisesRegex(EmbeddingStorageError, "返回列"):
                self.store.list_source("p", "code:one", "a.py")
        self.cursor.description = [SimpleNamespace(name=name) for name in self.columns]
        self.cursor.fetchall.return_value = [("p",)]
        with self.assertRaisesRegex(EmbeddingStorageError, "字段数量"):
            self.store.list_source("p", "code:one", "a.py")

    def test_query_failure_is_not_empty_result_and_does_not_expose_secrets(self):
        self.conn.execute.side_effect = RuntimeError("fake-test-password")
        with self.assertRaises(EmbeddingStorageError) as error:
            self.store.list_source("p", "code:one", "a.py")
        self.assertNotIn("fake-test-password", str(error.exception))
        self.assertIn("RuntimeError", str(error.exception))
