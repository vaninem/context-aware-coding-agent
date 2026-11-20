"""查询入口只编码问题、检查就绪状态并委托数据库检索。"""

import unittest
from unittest.mock import Mock

from context_lab.embeddings import EmbeddingConfig, IncrementalEmbeddingIndex, IndexStatus, PostgresEmbeddingStore
from context_lab.embeddings.errors import EmbeddingRequestError, EmbeddingStorageError, IndexNotReady


class EmbeddingSearchTests(unittest.TestCase):
    def setUp(self):
        self.store = Mock(spec=PostgresEmbeddingStore)
        self.store.status.return_value = IndexStatus(ready=2)
        self.store.search_ready.return_value = [object()]
        self.client = Mock()
        self.client.embed.return_value = [[1.0, 2.0]]
        self.index = IncrementalEmbeddingIndex(EmbeddingConfig(), client=self.client, store=self.store)

    def test_only_query_is_encoded_and_database_hits_are_returned(self):
        hits = self.index.search("query text", "code:one", limit=3, source_id="a.py")
        self.assertIs(hits, self.store.search_ready.return_value)
        self.store.status.assert_called_once_with(self.index.profile_id, "code:one")
        self.client.embed.assert_called_once_with(["query text"])
        self.store.search_ready.assert_called_once_with(self.index.profile_id, "code:one", [1.0, 2.0], 3, "a.py")
        for name in ("list_ready", "claim_jobs", "commit_source", "get_cached", "initialize"):
            getattr(self.store, name).assert_not_called()

    def test_empty_namespace_returns_without_model_request(self):
        self.store.status.return_value = IndexStatus()
        self.assertEqual(self.index.search("query", "code:one"), [])
        self.client.embed.assert_not_called()
        self.store.search_ready.assert_not_called()

    def test_any_unready_state_blocks_before_model_even_when_no_ready_chunks(self):
        for status in (IndexStatus(pending=1), IndexStatus(running=1), IndexStatus(failed=1),
                       IndexStatus(ready=5, pending=1)):
            self.store.status.return_value = status
            with self.subTest(status=status), self.assertRaises(IndexNotReady):
                self.index.search("query", "code:one")
        self.client.embed.assert_not_called()
        self.store.search_ready.assert_not_called()

    def test_source_filter_still_uses_namespace_readiness_policy(self):
        self.store.status.return_value = IndexStatus(ready=2, failed=1)
        with self.assertRaises(IndexNotReady):
            self.index.search("query", "code:one", source_id="a.py")
        self.client.embed.assert_not_called()

    def test_invalid_arguments_stop_before_storage(self):
        for args in (("", "n"), (None, "n"), ("query", " "), ("query", "n", 0),
                     ("query", "n", True), ("query", "n", 3, " ")):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.index.search(*args)
        self.store.status.assert_not_called()
        self.client.embed.assert_not_called()

    def test_wrong_response_count_is_not_silently_truncated(self):
        for vectors in ([], [[1.0], [2.0]], None):
            self.client.embed.return_value = vectors
            with self.subTest(vectors=vectors), self.assertRaises(EmbeddingRequestError):
                self.index.search("query", "n")
        self.store.search_ready.assert_not_called()

    def test_model_error_is_propagated_without_indexing(self):
        error = EmbeddingRequestError("timeout", "timeout", True)
        self.client.embed.side_effect = error
        with self.assertRaises(EmbeddingRequestError) as caught:
            self.index.search("query", "n")
        self.assertIs(caught.exception, error)
        self.store.search_ready.assert_not_called()
        self.store.fail_job.assert_not_called()

    def test_storage_errors_are_not_empty_hits(self):
        self.store.search_ready.side_effect = EmbeddingStorageError("test database failure")
        with self.assertRaises(EmbeddingStorageError):
            self.index.search("query", "n")
