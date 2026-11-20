"""失败处理的模拟事务测试，不连接真实数据库。"""

import hashlib
import unittest
from unittest.mock import MagicMock, patch

from context_lab.embeddings import EmbeddingJob, IndexedChunk, PostgresEmbeddingStore
from context_lab.embeddings.errors import EmbeddingStorageError


class EmbeddingFailTests(unittest.TestCase):
    def setUp(self):
        self.store = PostgresEmbeddingStore("postgresql://invalid.example/test", schema="test_fail")
        self.chunk = IndexedChunk("p", "code:one", "a.py", "a", "text",
                                  hashlib.sha256(b"text").hexdigest(), state="running")
        self.job = EmbeddingJob("job-a", self.chunk, 2, "lease-a", 123.0)
        self.conn = MagicMock()
        self.conn.__enter__.return_value = self.conn
        self.conn.__exit__.return_value = False
        self.transaction = self.conn.transaction.return_value
        self.transaction.__exit__.return_value = False
        patcher = patch.object(self.store, "connect", return_value=self.conn)
        self.connect = patcher.start()
        self.addCleanup(patcher.stop)
        self.current = ("running", self.chunk.content_hash)
        self.lease = ("lease-a",)
        self.job_updated = (self.job.id,)
        self.chunk_updated = (self.chunk.chunk_id,)
        self.chunk_error = False
        self.calls = []
        self.conn.execute.side_effect = self.execute

    def execute(self, query, params):
        text = query if isinstance(query, str) else query.as_string()
        self.calls.append((text, params))
        result = MagicMock()
        if "SELECT state, content_hash" in text:
            result.fetchone.return_value = self.current
        elif "SELECT lease_token" in text:
            result.fetchone.return_value = self.lease
        elif "RETURNING job_id" in text:
            result.fetchone.return_value = self.job_updated
        elif "RETURNING chunk_id" in text:
            if self.chunk_error:
                raise RuntimeError("fake-test-password")
            result.fetchone.return_value = self.chunk_updated
        return result

    def updates(self):
        return [(query, params) for query, params in self.calls if query.lstrip().startswith("UPDATE")]

    def test_retry_and_permanent_failure_update_only_current_job_and_chunk(self):
        for retryable, state in ((True, "pending"), (False, "failed")):
            self.calls.clear()
            with self.subTest(retryable=retryable):
                self.assertIsNone(self.store.fail_job(self.job, "timeout", retryable))
                job_update, chunk_update = self.updates()
                self.assertIn("WHERE job_id = %s", job_update[0])
                self.assertIn("AND lease_token = %s AND state = 'running'", job_update[0])
                self.assertIn("lease_expires_at > clock_timestamp()", job_update[0])
                self.assertIn("CASE WHEN %s", job_update[0])
                self.assertIn("INTERVAL '5 seconds' ELSE NULL END", job_update[0])
                self.assertIn("lease_token = NULL, lease_expires_at = NULL", job_update[0])
                self.assertNotIn("attempts =", job_update[0])
                self.assertEqual(job_update[1], (state, retryable, "timeout", "job-a", "p", "code:one",
                                                "a.py", "a", self.chunk.content_hash, "lease-a"))
                self.assertEqual(chunk_update[1], (state, "p", "code:one", "a.py", "a", self.chunk.content_hash))
                self.assertTrue(all("vectors" not in query for query, _ in self.calls))

    def test_missing_changed_or_finished_chunk_causes_no_updates(self):
        for row in (None, ("running", "new-hash"), ("ready", self.chunk.content_hash)):
            self.calls.clear()
            self.current = row
            with self.subTest(row=row):
                self.store.fail_job(self.job, "timeout", True)
                self.assertEqual(self.updates(), [])

    def test_missing_expired_or_reclaimed_job_causes_no_updates(self):
        for row in (None, ("other-lease",)):
            self.calls.clear()
            self.lease = row
            with self.subTest(row=row):
                self.store.fail_job(self.job, "timeout", True)
                self.assertEqual(self.updates(), [])

    def test_expiry_during_lock_wait_does_not_update_chunk(self):
        self.job_updated = None
        self.store.fail_job(self.job, "timeout", True)
        self.assertEqual(len(self.updates()), 1)

    def test_missing_chunk_during_update_aborts_transaction(self):
        self.chunk_updated = None
        with self.assertRaises(EmbeddingStorageError):
            self.store.fail_job(self.job, "timeout", True)
        self.assertIs(self.transaction.__exit__.call_args.args[0], EmbeddingStorageError)

    def test_database_failure_aborts_transaction_and_hides_details(self):
        self.chunk_error = True
        with self.assertRaises(EmbeddingStorageError) as error:
            self.store.fail_job(self.job, "timeout", True)
        self.assertNotIn("fake-test-password", str(error.exception))
        self.assertIs(self.transaction.__exit__.call_args.args[0], RuntimeError)

    def test_bad_error_code_or_retry_flag_never_connects(self):
        for code, retryable in (("", True), ("raw error text", True), ("x" * 65, True),
                                (None, True), ("timeout", "true")):
            with self.subTest(code=code), self.assertRaises(ValueError):
                self.store.fail_job(self.job, code, retryable)
        self.connect.assert_not_called()
