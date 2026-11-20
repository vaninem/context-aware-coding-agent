"""初始化接线测试：检查 SQL 组合、执行顺序和异常边界，不代替真实建表验收。"""

import unittest
from unittest.mock import MagicMock, Mock, patch

from context_lab.embeddings import PostgresEmbeddingStore
from context_lab.embeddings.errors import EmbeddingStorageError


class EmbeddingInitializeTests(unittest.TestCase):
    def setUp(self):
        self.conn = MagicMock()
        self.conn.__enter__.return_value = self.conn
        self.conn.__exit__.return_value = False
        self.factory = Mock(return_value=self.conn)
        self.store = PostgresEmbeddingStore(
            "postgresql://test:fake-test-password@invalid.example/test",
            schema="test_embeddings_init", connection_factory=self.factory,
        )
        self.statements = []
        def execute(query):
            self.statements.append(query if isinstance(query, str) else query.as_string())
        self.conn.execute.side_effect = execute

    def test_bootstrap_precedes_registration_and_all_tables_precede_indexes(self):
        def register(conn):
            self.assertIs(conn, self.conn)
            self.assertEqual(self.statements, ["CREATE EXTENSION IF NOT EXISTS vector"])
        with patch("context_lab.embeddings.store.register_vector", side_effect=register), patch.object(self.store, "connect") as connect:
            self.assertIsNone(self.store.initialize())
        connect.assert_not_called()
        self.factory.assert_called_once_with(self.store.database_url, connect_timeout=10)
        self.assertIn('CREATE SCHEMA IF NOT EXISTS "test_embeddings_init"', self.statements[1])
        for statement, name in zip(self.statements[2:6], ("profiles", "vectors", "chunks", "jobs")):
            self.assertIn(f'CREATE TABLE IF NOT EXISTS "test_embeddings_init"."{name}"', statement)
        self.assertEqual(len(self.statements), 10)
        self.assertTrue(all("CREATE INDEX IF NOT EXISTS" in statement for statement in self.statements[6:]))
        self.conn.__exit__.assert_called_once_with(None, None, None)

    def test_schema_captures_cache_identity_chunk_identity_and_job_leases(self):
        with patch("context_lab.embeddings.store.register_vector"):
            self.store.initialize()
        vectors, chunks, jobs = self.statements[3:6]
        self.assertIn("embedding VECTOR NOT NULL", vectors)
        self.assertIn("PRIMARY KEY (profile_id, content_hash)", vectors)
        self.assertIn('REFERENCES "test_embeddings_init"."profiles"', vectors)
        self.assertIn("content_hash TEXT NOT NULL", chunks)
        self.assertIn("metadata JSONB NOT NULL DEFAULT '{}'::jsonb", chunks)
        self.assertIn("PRIMARY KEY (profile_id, namespace, source_id, chunk_id)", chunks)
        self.assertIn("UNIQUE (profile_id, namespace, source_id, chunk_id, content_hash)", jobs)
        self.assertIn("lease_expires_at TIMESTAMPTZ", jobs)
        self.assertIn("ON DELETE CASCADE", jobs)
        self.assertIn("CHECK ((lease_token IS NULL) = (lease_expires_at IS NULL))", jobs)
        self.assertTrue(all("DROP " not in statement.upper() for statement in self.statements))
        self.assertTrue(all("HNSW" not in statement.upper() for statement in self.statements))

    def test_repeated_initialize_only_uses_if_not_exists_ddl(self):
        with patch("context_lab.embeddings.store.register_vector"):
            self.store.initialize()
            first = list(self.statements)
            self.store.initialize()
        self.assertEqual(self.statements, first + first)
        self.assertTrue(all("IF NOT EXISTS" in statement for statement in first))

    def test_failure_reaches_connection_context_and_is_redacted(self):
        self.conn.execute.side_effect = RuntimeError("fake-test-password")
        with patch("context_lab.embeddings.store.register_vector") as register:
            with self.assertRaises(EmbeddingStorageError) as error:
                self.store.initialize()
        register.assert_not_called()
        self.assertEqual(self.conn.__exit__.call_args.args[0], RuntimeError)
        self.assertIn("RuntimeError", str(error.exception))
        self.assertNotIn("fake-test-password", str(error.exception))
        self.assertTrue(error.exception.__suppress_context__)

    def test_default_factory_uses_psycopg_connect(self):
        self.store.connection_factory = None
        with patch("psycopg.connect", return_value=self.conn) as connect, patch("context_lab.embeddings.store.register_vector"):
            self.store.initialize()
        connect.assert_called_once_with(self.store.database_url, connect_timeout=10)

    def test_invalid_configuration_never_connects(self):
        cases = [
            {"database_url": ""}, {"connect_timeout": 0}, {"connect_timeout": True},
            {"schema": "../bad"}, {"schema": "a" * 64}, {"schema": "x.y"},
        ]
        for change in cases:
            settings = {"database_url": "postgresql://invalid.example/test", "connection_factory": self.factory, **change}
            with self.subTest(change=change), self.assertRaises(ValueError):
                PostgresEmbeddingStore(**settings).initialize()
        self.factory.assert_not_called()
