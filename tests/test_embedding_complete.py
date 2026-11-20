"""结果提交的模拟事务测试；不访问模型或真实数据库。"""

from dataclasses import replace
import hashlib
import unittest
from unittest.mock import MagicMock, patch

from context_lab.embeddings import EmbeddingJob, IndexedChunk, PostgresEmbeddingStore
from context_lab.embeddings.errors import EmbeddingStorageError


class EmbeddingCompleteTests(unittest.TestCase):
    def setUp(self):
        self.store = PostgresEmbeddingStore("postgresql://invalid.example/test", schema="test_complete")
        self.chunk = IndexedChunk("p", "code:one", "a.py", "a", "text",
                                  hashlib.sha256(b"text").hexdigest(), state="running")
        self.job = EmbeddingJob("job-a", self.chunk, 1, "lease-a", 123.0)
        self.conn = MagicMock()
        self.conn.__enter__.return_value = self.conn
        self.conn.__exit__.return_value = False
        self.transaction = self.conn.transaction.return_value
        self.transaction.__exit__.return_value = False
        patcher = patch.object(self.store, "connect", return_value=self.conn)
        self.connect = patcher.start()
        self.addCleanup(patcher.stop)
        self.current = (self.chunk.content_hash, "running")
        self.job_row = (self.chunk.content_hash, "lease-a")
        self.profile = (None,)
        self.finish_job = (self.job.id,)
        self.finish_chunk = (self.chunk.chunk_id,)
        self.vector_error = False
        self.calls = []
        self.conn.execute.side_effect = self.execute

    def execute(self, query, params):
        text = query if isinstance(query, str) else query.as_string()
        self.calls.append((text, params))
        result = MagicMock()
        if "SELECT content_hash, state" in text:
            result.fetchone.return_value = self.current
        elif "SELECT content_hash, lease_token" in text:
            result.fetchone.return_value = self.job_row
        elif "SELECT dimensions" in text:
            result.fetchone.return_value = self.profile
        elif "RETURNING job_id" in text:
            result.fetchone.return_value = self.finish_job
        elif "RETURNING chunk_id" in text:
            result.fetchone.return_value = self.finish_chunk
        elif 'INSERT INTO "test_complete"."vectors"' in text and self.vector_error:
            raise RuntimeError("fake-test-password")
        return result

    def writes(self):
        return [(query, params) for query, params in self.calls if query.lstrip().startswith(("UPDATE", "INSERT"))]

    def test_initial_dimension_cache_and_both_states_commit_together(self):
        self.assertTrue(self.store.complete_job(self.job, [1.0, 2.0]))
        writes = self.writes()
        self.assertEqual(len(writes), 4)
        self.assertIn("lease_token = NULL", writes[0][0])
        self.assertIn("lease_expires_at = NULL", writes[0][0])
        self.assertIn("lease_expires_at > clock_timestamp()", writes[0][0])
        self.assertEqual(writes[0][1], ("job-a", "p", "code:one", "a.py", "a", self.chunk.content_hash, "lease-a"))
        self.assertEqual(writes[1][1], (2, "p"))
        self.assertEqual(writes[2][1][2].to_list(), [1.0, 2.0])
        self.assertIn("DO NOTHING", writes[2][0])
        self.assertEqual(writes[3][1], ("p", "code:one", "a.py", "a", self.chunk.content_hash))
        self.transaction.__exit__.assert_called_once_with(None, None, None)

    def test_existing_dimension_is_checked_without_rewriting(self):
        self.profile = (2,)
        self.assertTrue(self.store.complete_job(self.job, [1.0, 2.0]))
        self.assertEqual(len(self.writes()), 3)
        dimension_select = next(query for query, _ in self.calls if "SELECT dimensions" in query)
        self.assertIn("FOR UPDATE", dimension_select)

    def test_deleted_changed_or_finished_chunk_is_rejected_without_writes(self):
        for current in (None, ("new-hash", "running"), (self.chunk.content_hash, "ready")):
            self.calls.clear()
            self.current = current
            with self.subTest(current=current):
                self.assertFalse(self.store.complete_job(self.job, [1.0]))
                self.assertEqual(self.writes(), [])

    def test_missing_expired_or_reclaimed_job_is_rejected_without_writes(self):
        for row in (None, (self.chunk.content_hash, "new-lease"), ("new-hash", "lease-a")):
            self.calls.clear()
            self.job_row = row
            with self.subTest(row=row):
                self.assertFalse(self.store.complete_job(self.job, [1.0]))
                self.assertEqual(self.writes(), [])

    def test_dimension_errors_abort_before_writes(self):
        for profile in (None, (3,)):
            self.calls.clear()
            self.profile = profile
            with self.subTest(profile=profile), self.assertRaises(EmbeddingStorageError):
                self.store.complete_job(self.job, [1.0, 2.0])
            self.assertEqual(self.writes(), [])

    def test_lease_expiring_while_waiting_for_dimension_lock_does_not_save_vector(self):
        self.finish_job = None
        self.assertFalse(self.store.complete_job(self.job, [1.0, 2.0]))
        self.assertEqual(len(self.writes()), 1)
        self.assertFalse(any("INSERT INTO" in query for query, _ in self.calls))

    def test_late_chunk_failure_aborts_whole_transaction(self):
        self.finish_chunk = None
        with self.assertRaises(EmbeddingStorageError):
            self.store.complete_job(self.job, [1.0, 2.0])
        self.assertEqual(len(self.writes()), 4)
        self.assertIs(self.transaction.__exit__.call_args.args[0], EmbeddingStorageError)

    def test_vector_write_failure_aborts_and_redacts_error(self):
        self.vector_error = True
        with self.assertRaises(EmbeddingStorageError) as error:
            self.store.complete_job(self.job, [1.0, 2.0])
        self.assertNotIn("fake-test-password", str(error.exception))
        self.assertIs(self.transaction.__exit__.call_args.args[0], RuntimeError)

    def test_invalid_vector_or_missing_lease_never_connects(self):
        for vector in ([], [0.0, 0.0], [float("nan")], [float("inf")], [True], [[1.0]], [1e100], [1e-100]):
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                self.store.complete_job(self.job, vector)
        with self.assertRaises(ValueError):
            self.store.complete_job(replace(self.job, lease_token=None), [1.0])
        self.connect.assert_not_called()
