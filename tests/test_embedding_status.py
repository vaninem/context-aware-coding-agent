"""状态统计的模拟数据库测试，不访问真实数据库。"""

import unittest
from unittest.mock import MagicMock, patch

from context_lab.embeddings import IndexStatus, PostgresEmbeddingStore
from context_lab.embeddings.errors import EmbeddingStorageError


class EmbeddingStatusTests(unittest.TestCase):
    def setUp(self):
        self.store = PostgresEmbeddingStore("postgresql://invalid.example/test", schema="test_status")
        self.conn = MagicMock()
        self.conn.__enter__.return_value = self.conn
        self.conn.__exit__.return_value = False
        patcher = patch.object(self.store, "connect", return_value=self.conn)
        self.connect = patcher.start()
        self.addCleanup(patcher.stop)

    def test_scoped_counts_and_missing_states_default_to_zero(self):
        self.conn.execute.return_value = iter([("pending", 3), ("ready", 5)])
        self.assertEqual(self.store.status("p", "code:one"), IndexStatus(pending=3, ready=5))
        query, params = self.conn.execute.call_args.args
        self.assertEqual(params, ("p", "code:one"))
        text = query.as_string()
        self.assertIn('FROM "test_status"."chunks"', text)
        self.assertIn("WHERE profile_id = %s AND namespace = %s", text)
        self.assertIn("GROUP BY state", text)

    def test_empty_database_returns_all_zero(self):
        self.conn.execute.return_value = iter([])
        self.assertEqual(self.store.status("p", "n"), IndexStatus())

    def test_invalid_scope_does_not_connect(self):
        for args in (("", "n"), ("p", " "), (None, "n")):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.store.status(*args)
        self.connect.assert_not_called()

    def test_corrupt_counts_raise(self):
        for rows in ([("unknown", 1)], [("ready", -1)], [("ready", "2")]):
            self.conn.execute.return_value = iter(rows)
            with self.subTest(rows=rows), self.assertRaises(EmbeddingStorageError):
                self.store.status("p", "n")

    def test_database_failure_is_not_empty_status_and_hides_secrets(self):
        self.conn.execute.side_effect = RuntimeError("fake-test-password")
        with self.assertRaises(EmbeddingStorageError) as error:
            self.store.status("p", "n")
        self.assertNotIn("fake-test-password", str(error.exception))
        self.assertIn("RuntimeError", str(error.exception))
