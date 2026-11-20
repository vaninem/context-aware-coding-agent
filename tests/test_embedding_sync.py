"""E12：模拟存储检查差异计划，不连接 PostgreSQL、不调用模型。"""

import hashlib
import unittest
from unittest.mock import Mock

from context_lab.embeddings import (
    EmbeddingChunk, EmbeddingConfig, IncrementalEmbeddingIndex, IndexedChunk,
    PostgresEmbeddingStore, SyncReport,
)


class EmbeddingSyncTests(unittest.TestCase):
    def setUp(self):
        self.store = Mock(spec=PostgresEmbeddingStore)
        self.store.list_source.return_value = []
        self.store.get_cached.return_value = None
        self.client = Mock()
        self.client.embed.side_effect = AssertionError("同步不能调用模型")
        self.index = IncrementalEmbeddingIndex(EmbeddingConfig(), client=self.client, store=self.store)

    def old(self, chunk_id, text, state="ready", metadata=None):
        return IndexedChunk(
            self.index.profile_id, "code:one", "a.py", chunk_id, text,
            hashlib.sha256(text.encode("utf-8")).hexdigest(),
            metadata or {}, state,
        )

    def committed(self):
        self.store.commit_source.assert_called_once()
        profile, namespace, source, chunks, jobs = self.store.commit_source.call_args.args
        self.assertEqual((profile, namespace, source), (self.index.profile_id, "code:one", "a.py"))
        self.client.embed.assert_not_called()
        return chunks, jobs

    def test_complete_snapshot_mixes_unchanged_updated_created_deleted_and_cached(self):
        old = self.old("a", "same", metadata={"line": 1})
        self.store.list_source.return_value = [old, self.old("b", "before"), self.old("c", "deleted")]
        cached_hash = self.index._content_hash("cached")
        self.store.get_cached.side_effect = lambda profile, digest: [1.0, 1.0] if digest == cached_hash else None
        report = self.index.sync_source("code:one", "a.py", [
            EmbeddingChunk("a", "same", {"line": 99}),
            EmbeddingChunk("b", "after"), EmbeddingChunk("d", "cached"),
        ])
        self.assertEqual(report, SyncReport(created=1, updated=1, unchanged=1, deleted=1, cache_hits=1, queued=1))
        chunks, jobs = self.committed()
        self.assertEqual([chunk.chunk_id for chunk in chunks], ["a", "b", "d"])
        self.assertEqual([chunk.state for chunk in chunks], ["ready", "pending", "ready"])
        self.assertEqual(chunks[0].metadata, {"line": 99})
        self.assertEqual(old.metadata, {"line": 1})
        self.assertEqual(chunks[1].text, "after")
        self.assertEqual(chunks[1].content_hash, self.index._content_hash("after"))
        self.assertEqual(len(jobs), 1)
        self.assertIs(jobs[0].chunk, chunks[1])
        self.assertTrue(jobs[0].id.startswith("job_"))

    def test_renamed_chunk_can_reuse_content_cache(self):
        self.store.list_source.return_value = [self.old("old-id", "same")]
        self.store.get_cached.return_value = [1.0, 1.0]
        report = self.index.sync_source("code:one", "a.py", [EmbeddingChunk("new-id", "same")])
        self.assertEqual(report, SyncReport(created=1, deleted=1, cache_hits=1))
        chunks, jobs = self.committed()
        self.assertEqual(chunks[0].chunk_id, "new-id")
        self.assertEqual(chunks[0].state, "ready")
        self.assertEqual(jobs, [])

    def test_unchanged_blocks_keep_states_without_requeuing(self):
        states = ["ready", "pending", "running", "failed"]
        self.store.list_source.return_value = [self.old(state, state, state) for state in states]
        report = self.index.sync_source("code:one", "a.py", [EmbeddingChunk(state, state) for state in states])
        self.assertEqual(report, SyncReport(unchanged=4))
        chunks, jobs = self.committed()
        self.assertEqual([chunk.state for chunk in chunks], states)
        self.assertEqual(jobs, [])
        self.store.get_cached.assert_not_called()

    def test_empty_snapshot_deletes_only_requested_source(self):
        self.store.list_source.return_value = [self.old("a", "one"), self.old("b", "two")]
        report = self.index.delete_source("code:one", "a.py")
        self.assertEqual(report, SyncReport(deleted=2))
        self.assertEqual(self.committed(), ([], []))
        self.store.list_source.assert_called_once_with(self.index.profile_id, "code:one", "a.py")

    def test_metadata_is_detached_even_when_nested(self):
        metadata = {"location": {"line": 1}, "tags": ["code"]}
        self.index.sync_source("code:one", "a.py", [EmbeddingChunk("a", "text", metadata)])
        chunks, _ = self.committed()
        metadata["location"]["line"] = 100
        metadata["tags"].append("changed")
        self.assertEqual(chunks[0].metadata, {"location": {"line": 1}, "tags": ["code"]})

    def test_commit_failure_is_propagated(self):
        self.store.commit_source.side_effect = RuntimeError("test commit failure")
        with self.assertRaisesRegex(RuntimeError, "test commit failure"):
            self.index.sync_source("code:one", "a.py", [EmbeddingChunk("a", "text")])
        self.client.embed.assert_not_called()

    def test_cache_failure_does_not_commit(self):
        self.store.get_cached.side_effect = RuntimeError("test cache failure")
        with self.assertRaisesRegex(RuntimeError, "test cache failure"):
            self.index.sync_source("code:one", "a.py", [EmbeddingChunk("a", "text")])
        self.store.commit_source.assert_not_called()

    def test_invalid_snapshot_is_rejected_before_storage_access(self):
        invalid = [
            ("", "a.py", []), ("code:one", " ", []), ("code:one", "a.py", None),
            ("code:one", "a.py", [None]),
            ("code:one", "a.py", [EmbeddingChunk("", "text")]),
            ("code:one", "a.py", [EmbeddingChunk("a", " ")]),
            ("code:one", "a.py", [EmbeddingChunk("a", "text"), EmbeddingChunk("a", "other")]),
            ("code:one", "a.py", [EmbeddingChunk("a", "text", {"x": float("nan")})]),
            ("code:one", "a.py", [EmbeddingChunk("a", "text", {"x": object()})]),
        ]
        for namespace, source, chunks in invalid:
            with self.subTest(namespace=namespace, source=source), self.assertRaises(ValueError):
                self.index.sync_source(namespace, source, chunks)
        self.store.list_source.assert_not_called()
        self.store.commit_source.assert_not_called()
