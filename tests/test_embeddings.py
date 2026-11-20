"""框架测试应通过；业务测试初始抛 E11–E15，不能视为实现已完成。"""

import hashlib
import os
from dataclasses import replace
from uuid import uuid4

from context_lab.embeddings import (
    EmbeddingChunk, EmbeddingConfig, HTTPEmbeddingClient, IncrementalEmbeddingIndex,
    PostgresEmbeddingStore, SyncReport,
)
from tests.support import SandboxTestCase


class FakeEmbeddingClient:
    """只检查调用记录与接线，三个固定数值不代表真实语义效果。"""
    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [[float(len(text)), float(sum(map(ord, text)) % 11), 1.0] for text in texts]


class EmbeddingScaffoldTests(SandboxTestCase):
    def test_env_loading_is_explicit_and_does_not_expose_key(self):
        env = self.tmp / ".env"
        env.write_text(
            '# comment\nEMBEDDING_BASE_URL = https://invalid.example/v1/\n'
            'EMBEDDING_MODEL = test-model\nEMBEDDING_API_KEY = "fake-test-secret"\n'
            'DATABASE_URL=postgresql://lab:fake-db-secret@127.0.0.1:55432/test\n'
            'EMBEDDING_DB_SCHEMA=test_vectors\n', encoding="utf-8",
        )
        config = EmbeddingConfig.from_env(env, environ={})
        self.assertTrue(config.enabled)
        self.assertEqual(config.base_url, "https://invalid.example/v1")
        self.assertEqual(config.model, "test-model")
        self.assertNotIn("fake-test-secret", repr(config))
        self.assertNotIn("fake-db-secret", repr(config))
        self.assertTrue(config.database_configured)
        self.assertEqual(config.database_schema, "test_vectors")

    def test_environment_overrides_file_and_missing_file_is_offline(self):
        config = EmbeddingConfig.from_env(self.tmp / "absent.env", environ={})
        self.assertFalse(config.enabled)
        self.assertFalse(config.database_configured)
        env = self.tmp / ".env"
        env.write_text('EMBEDDING_MODEL=file-model\n', encoding="utf-8")
        overridden = EmbeddingConfig.from_env(env, environ={"EMBEDDING_MODEL": "env-model"})
        self.assertEqual(overridden.model, "env-model")

    def test_profile_changes_with_model_but_not_api_key(self):
        config = EmbeddingConfig(base_url="https://invalid.example/v1", model="one", api_key="a")
        self.assertEqual(config.profile_id, replace(config, api_key="b").profile_id)
        self.assertEqual(config.profile_id, replace(config, database_url="postgresql://other/db").profile_id)
        for change in ({"model": "two"}, {"revision": "v2"}, {"expected_dimensions": 3}):
            self.assertNotEqual(config.profile_id, replace(config, **change).profile_id)

    def test_constructors_only_save_dependencies(self):
        config = EmbeddingConfig(database_url="postgresql://lab:fake-db-secret@127.0.0.1/test")
        client = FakeEmbeddingClient()
        connections = []
        factory = lambda *args, **kwargs: connections.append(True)
        store = PostgresEmbeddingStore(config.database_url, connection_factory=factory)
        index = IncrementalEmbeddingIndex(config, client=client, store=store)
        self.assertIs(index.client, client)
        self.assertIs(index.store, store)
        self.assertEqual(index.profile_id, config.profile_id)
        self.assertEqual(client.calls, [])
        self.assertEqual(connections, [])
        self.assertEqual(store.database_url, config.database_url)
        self.assertIs(store.connection_factory, factory)

    def test_default_store_is_postgres_without_eager_connection(self):
        config = EmbeddingConfig(database_schema="agent_embeddings")
        index = IncrementalEmbeddingIndex(config)
        self.assertIsInstance(index.store, PostgresEmbeddingStore)
        self.assertEqual(index.store.schema, "agent_embeddings")
        self.assertIsNone(index.store.connection_factory)

    def test_schema_identifier_validation(self):
        for name in ("public;DROP TABLE x", "../bad", "a.b", "UPPER", ""):
            with self.subTest(name=name), self.assertRaises(ValueError):
                EmbeddingConfig(database_schema=name)

    def test_default_client_does_not_connect(self):
        client = HTTPEmbeddingClient(EmbeddingConfig(base_url="https://invalid.example/v1"))
        self.assertEqual(client.calls, 0)
        self.assertEqual(client.endpoint, "https://invalid.example/v1/embeddings")
        self.assertIsNone(client.transport)

    def test_mutable_defaults_are_independent(self):
        first, second = EmbeddingChunk("a", "text"), EmbeddingChunk("b", "text")
        first.metadata["line"] = 1
        self.assertEqual(second.metadata, {})
        self.assertEqual(SyncReport().queued, 0)


class EmbeddingExerciseTests(SandboxTestCase):
    def make_index(self):
        config = EmbeddingConfig(
            base_url="https://invalid.example/v1", model="fake-test-model", api_key="fake-test-key",
            database_url=os.environ.get("TEST_DATABASE_URL", ""),
            database_schema="test_embeddings_" + uuid4().hex, expected_dimensions=3,
        )
        self.client = FakeEmbeddingClient()
        return IncrementalEmbeddingIndex(config, client=self.client)

    def initialize_index(self, index):
        if not index.config.database_configured:
            raise RuntimeError("数据库练习需显式设置 TEST_DATABASE_URL；不会使用真实业务配置或跳过测试")
        index.initialize()
        self.addCleanup(self.drop_test_schema, index.config)

    @staticmethod
    def drop_test_schema(config):
        import psycopg
        from psycopg import sql
        if not config.database_schema.startswith("test_embeddings_"):
            raise RuntimeError("拒绝清理非测试 schema")
        with psycopg.connect(config.database_url, connect_timeout=10) as connection:
            connection.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                sql.Identifier(config.database_schema),
            ))

    def test_e11_response_indices_are_used(self):
        client = HTTPEmbeddingClient(EmbeddingConfig(expected_dimensions=3))
        vectors = client._parse_response({"data": [
            {"index": 1, "embedding": [0.0, 1.0, 0.0]},
            {"index": 0, "embedding": [1.0, 0.0, 0.0]},
        ]}, expected_count=2)
        self.assertEqual(vectors, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    def test_e12_content_hash_is_stable(self):
        index = self.make_index()
        self.assertEqual(index._content_hash("中文"), hashlib.sha256("中文".encode()).hexdigest())

    def test_e13_e14_unchanged_and_changed_chunks(self):
        index = self.make_index()
        self.initialize_index(index)
        original = [EmbeddingChunk("a", "alpha"), EmbeddingChunk("b", "beta")]
        report = index.sync_source("code:one", "a.py", original)
        self.assertEqual(report.queued, 2)
        self.assertEqual(self.client.calls, [], "同步内容时不能请求嵌入")
        self.assertEqual(index.process_pending(), 2)
        self.assertCountEqual([text for batch in self.client.calls for text in batch], ["alpha", "beta"])
        self.client.calls.clear()
        report = index.sync_source("code:one", "a.py", original)
        self.assertEqual(report.unchanged, 2)
        self.assertEqual(report.queued, 0)
        self.assertEqual(index.process_pending(), 0)
        self.assertEqual(self.client.calls, [])
        index.sync_source("code:one", "a.py", [original[0], EmbeddingChunk("b", "changed")])
        self.assertEqual(index.process_pending(), 1)
        self.assertEqual(self.client.calls, [["changed"]])

    def test_e13_cached_content_reuses_vector_despite_new_chunk_id(self):
        index = self.make_index()
        self.initialize_index(index)
        index.sync_source("code:one", "a.py", [EmbeddingChunk("old", "same")])
        index.process_pending()
        self.client.calls.clear()
        report = index.sync_source("code:one", "a.py", [EmbeddingChunk("new", "same", {"line": 99})])
        self.assertEqual(report.cache_hits, 1)
        self.assertEqual(report.queued, 0)
        self.assertEqual(index.process_pending(), 0)
        ready = index.store.list_ready(index.profile_id, "code:one", "a.py")
        self.assertEqual([chunk.chunk_id for chunk, _ in ready], ["new"])
        self.assertEqual(self.client.calls, [])

    def test_e15_restart_search_only_embeds_query_and_delete_is_scoped(self):
        index = self.make_index()
        self.initialize_index(index)
        index.sync_source("memory:alice", "m1", [EmbeddingChunk("body", "apple")])
        index.sync_source("memory:bob", "m2", [EmbeddingChunk("body", "banana")])
        index.process_pending()
        restarted_client = FakeEmbeddingClient()
        restarted = IncrementalEmbeddingIndex(index.config, client=restarted_client)
        restarted.initialize()
        hits = restarted.search("apple", "memory:alice")
        self.assertTrue(hits)
        self.assertTrue(all(hit.chunk.namespace == "memory:alice" for hit in hits))
        self.assertEqual(restarted_client.calls, [["apple"]])
        restarted.delete_source("memory:alice", "m1")
        self.assertEqual(restarted.store.list_ready(index.profile_id, "memory:alice"), [])
        self.assertEqual(len(restarted.store.list_ready(index.profile_id, "memory:bob")), 1)
