"""就绪索引读取的模拟连接测试，不访问真实 PostgreSQL。"""

from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from pgvector import Vector

from context_lab.embeddings import IndexedChunk, PostgresEmbeddingStore
from context_lab.embeddings.errors import EmbeddingStorageError


class EmbeddingReadyTests(unittest.TestCase):
    def setUp(self):
        self.store = PostgresEmbeddingStore("postgresql://invalid.example/test", schema="test_ready")
        self.conn = MagicMock()
        self.conn.__enter__.return_value = self.conn
        self.conn.__exit__.return_value = False
        self.cursor = self.conn.execute.return_value
        self.columns = ["profile_id", "namespace", "source_id", "chunk_id", "text",
                        "content_hash", "metadata", "state", "embedding"]
        self.cursor.description = [SimpleNamespace(name=name) for name in self.columns]
        self.cursor.fetchall.return_value = []
        patcher = patch.object(self.store, "connect", return_value=self.conn)
        self.connect = patcher.start()
        self.addCleanup(patcher.stop)

    def record(self, **changes):
        return {"profile_id": "p", "namespace": "code:one", "source_id": "a.py", "chunk_id": "a",
                "text": "text", "content_hash": "hash-a", "metadata": {"location": {"line": 1}},
                "state": "ready", "embedding": np.array([1, 2], dtype=np.float32), **changes}

    def provide(self, record):
        self.cursor.fetchall.return_value = [tuple(record[name] for name in self.columns)]

    def test_joined_row_restores_chunk_vector_and_detaches_metadata(self):
        record = self.record()
        self.provide(record)
        chunk, vector = self.store.list_ready("p", "code:one")[0]
        self.assertIsInstance(chunk, IndexedChunk)
        self.assertEqual(vector, [1.0, 2.0])
        self.assertTrue(all(type(value) is float for value in vector))
        chunk.metadata["location"]["line"] = 9
        self.assertEqual(record["metadata"]["location"]["line"], 1)
        query, params = self.conn.execute.call_args.args
        text = query.as_string()
        self.assertIn('FROM "test_ready"."chunks" AS c', text)
        self.assertIn('JOIN "test_ready"."vectors" AS v', text)
        self.assertIn("c.profile_id = v.profile_id AND c.content_hash = v.content_hash", text)
        self.assertIn("c.state = 'ready'", text)
        self.assertIn("ORDER BY c.source_id, c.chunk_id", text)
        self.assertEqual(params, ("p", "code:one"))

    def test_optional_source_is_parameterized_in_database(self):
        source = "x'; DROP TABLE chunks; --"
        self.store.list_ready("p", "code:one", source)
        query, params = self.conn.execute.call_args.args
        self.assertIn("AND c.source_id = %s", query.as_string())
        self.assertNotIn(source, query.as_string())
        self.assertEqual(params, ("p", "code:one", source))

    def test_empty_result_is_empty_list(self):
        self.assertEqual(self.store.list_ready("p", "code:one"), [])

    def test_native_pgvector_object_is_supported(self):
        self.provide(self.record(embedding=Vector([1.0, 2.0])))
        self.assertEqual(self.store.list_ready("p", "code:one")[0][1], [1.0, 2.0])

    def test_column_order_is_mapped_by_name(self):
        self.columns.reverse()
        self.cursor.description = [SimpleNamespace(name=name) for name in self.columns]
        self.provide(self.record())
        chunk, vector = self.store.list_ready("p", "code:one")[0]
        self.assertEqual(chunk.chunk_id, "a")
        self.assertEqual(vector, [1.0, 2.0])

    def test_bad_or_out_of_scope_record_is_not_returned(self):
        for changes in ({"profile_id": "other"}, {"namespace": "other"}, {"source_id": "b.py"},
                        {"state": "pending"}, {"metadata": []}, {"text": " "}):
            self.provide(self.record(**changes))
            with self.subTest(changes=changes), self.assertRaises(EmbeddingStorageError):
                self.store.list_ready("p", "code:one", "a.py")

    def test_bad_vectors_raise_clear_error(self):
        for vector in (None, [], [0, 0], [True], [float("nan")], [float("inf")], "12", [[1, 2]], [10 ** 400]):
            self.provide(self.record(embedding=vector))
            with self.subTest(vector_type=type(vector).__name__), self.assertRaisesRegex(EmbeddingStorageError, "就绪向量损坏"):
                self.store.list_ready("p", "code:one")

    def test_missing_columns_or_short_row_are_rejected(self):
        self.cursor.description = None
        with self.assertRaisesRegex(EmbeddingStorageError, "返回列"):
            self.store.list_ready("p", "code:one")
        self.cursor.description = [SimpleNamespace(name=name) for name in self.columns]
        self.cursor.fetchall.return_value = [("p",)]
        with self.assertRaisesRegex(EmbeddingStorageError, "字段数量"):
            self.store.list_ready("p", "code:one")

    def test_bad_arguments_never_connect(self):
        for args in (("", "n"), ("p", None), ("p", "n", " "), ("p", "n", 1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.store.list_ready(*args)
        self.connect.assert_not_called()

    def test_query_failure_is_not_empty_result_and_hides_details(self):
        self.conn.execute.side_effect = RuntimeError("fake-test-password")
        with self.assertRaises(EmbeddingStorageError) as error:
            self.store.list_ready("p", "code:one")
        self.assertNotIn("fake-test-password", str(error.exception))
        self.assertIn("RuntimeError", str(error.exception))
