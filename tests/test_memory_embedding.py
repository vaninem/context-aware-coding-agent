"""M05 桥接验收：假模型/假存储，需已安装 embedding 依赖，但不访问网络。"""

import hashlib
from unittest.mock import Mock

from context_lab.embeddings import EmbeddingChunk, EmbeddingHit, IndexedChunk, IndexStatus
from context_lab.memory import MemoryIndexPending
from context_lab.memory.embedding_bridge import EmbeddingMemoryBridge
from tests.memory_support import NOW, record
from tests.support import SandboxTestCase


class EmbeddingBridgeM05Tests(SandboxTestCase):
    def setUp(self):
        super().setUp()
        self.index = Mock()
        self.index.profile_id = "profile"
        self.index.status.return_value = IndexStatus(ready=6)
        self.index.client.embed.return_value = [[1.0, 2.0]]
        self.bridge = EmbeddingMemoryBridge(self.index, "alice")

    def test_apply_uses_stable_source_and_body_chunk(self):
        item = record(valid_from=NOW, metadata={"user_id": "other", "memory_version": 99, "tag": "infra"})
        self.bridge.apply(item.id, item)
        namespace, source, chunks = self.index.sync_source.call_args.args
        self.assertEqual(namespace, "memory:alice")
        self.assertEqual(source, item.id)
        self.assertEqual(len(chunks), 1)
        self.assertIsInstance(chunks[0], EmbeddingChunk)
        self.assertEqual((chunks[0].id, chunks[0].text), ("body", item.content))
        self.assertEqual(chunks[0].metadata["user_id"], item.user_id)
        self.assertEqual(chunks[0].metadata["memory_version"], item.version)
        self.assertEqual(chunks[0].metadata["valid_from"], NOW.isoformat())
        self.assertEqual(chunks[0].metadata["tag"], "infra")
        self.index.client.embed.assert_not_called()

    def test_delete_and_stale_summary_remove_members_without_generating_vectors(self):
        self.bridge.apply("gone", None)
        self.index.sync_source.assert_called_with("memory:alice", "gone", [])
        stale = record("summary", layer="summary", summary_state="stale")
        self.bridge.apply(stale.id, stale)
        self.index.sync_source.assert_called_with("memory:alice", stale.id, [])
        self.index.process_pending.assert_not_called()

    def test_apply_rejects_mismatched_identity_before_sync(self):
        other_users_summary = record(user_id="bob", layer="summary", summary_state="stale")
        for memory_id, item in (("different", record()), (other_users_summary.id, other_users_summary)):
            with self.subTest(memory_id=memory_id, user_id=item.user_id), self.assertRaises(ValueError):
                self.bridge.apply(memory_id, item)
        self.index.sync_source.assert_not_called()

    def test_search_expands_window_after_filtering_and_embeds_query_once(self):
        valid = record("allowed", "valid fact")
        hits = []
        for i in range(5):
            chunk = IndexedChunk("profile", "memory:alice", f"expired-{i}", "body", "old",
                                 hashlib.sha256(b"old").hexdigest(), {"memory_version": 1}, "ready")
            hits.append(EmbeddingHit(chunk, 0.99 - i * 0.01))
        chunk = IndexedChunk("profile", "memory:alice", valid.id, "body", valid.content,
                             hashlib.sha256(valid.content.encode()).hexdigest(),
                             {"memory_version": valid.version, "user_id": "alice"}, "ready")
        hits.append(EmbeddingHit(chunk, 0.5))
        def database_search(*args, **kwargs):
            size = kwargs.get("limit", args[3] if len(args) > 3 else 5)
            return hits[:size]
        self.index.store.search_ready.side_effect = database_search
        results = self.bridge.search("query", [valid], 1)
        self.assertEqual([hit.memory.id for hit in results], [valid.id])
        self.assertGreaterEqual(self.index.store.search_ready.call_count, 2)
        self.index.client.embed.assert_called_once_with(["query"])

    def test_expanded_window_keeps_unique_candidates_and_raw_scores(self):
        first = record("first", "first body")
        last = record("last", "last body")
        excluded = [record(f"excluded-{i}", "excluded body") for i in range(7)]
        ranked = [first, *excluded, last]
        scores = [0.95] + [0.8] * len(excluded) + [0.35]
        hits = [
            EmbeddingHit(IndexedChunk(
                "profile", "memory:alice", item.id, "body", item.content,
                hashlib.sha256(item.content.encode("utf-8")).hexdigest(),
                {"memory_version": item.version, "user_id": item.user_id}, "ready",
            ), score)
            for item, score in zip(ranked, scores)
        ]
        self.index.status.return_value = IndexStatus(ready=len(hits))
        self.index.store.search_ready.side_effect = lambda *args, limit: hits[:limit]

        results = self.bridge.search("query", [first, last], 2)

        self.assertEqual([call.kwargs["limit"] for call in self.index.store.search_ready.call_args_list], [8, 9])
        self.assertEqual([hit.memory.id for hit in results], [first.id, last.id])
        self.assertEqual([(hit.score, hit.vector_score) for hit in results], [(0.95, 0.95), (0.35, 0.35)])
        self.index.client.embed.assert_called_once_with(["query"])

    def test_old_body_vector_is_not_returned_as_new_business_record(self):
        current = record("m1", "new body", version=2)
        stale = IndexedChunk("profile", "memory:alice", "m1", "body", "old body",
                             hashlib.sha256(b"old body").hexdigest(), {"memory_version": 1}, "ready")
        self.index.status.return_value = IndexStatus(ready=1)
        self.index.store.search_ready.return_value = [EmbeddingHit(stale, 0.9)]
        with self.assertRaises(MemoryIndexPending):
            self.bridge.search("query", [current], 1)
