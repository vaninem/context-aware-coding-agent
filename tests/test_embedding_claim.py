"""领取任务的模拟连接测试；真实锁竞争需要 PostgreSQL 集成验收。"""

from datetime import datetime, timezone
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from context_lab.embeddings import PostgresEmbeddingStore
from context_lab.embeddings.errors import EmbeddingStorageError


class EmbeddingClaimTests(unittest.TestCase):
    def setUp(self):
        self.store = PostgresEmbeddingStore("postgresql://invalid.example/test", schema="test_claim")
        self.conn = MagicMock()
        self.conn.__enter__.return_value = self.conn
        self.conn.__exit__.return_value = False
        self.transaction = self.conn.transaction.return_value
        self.transaction.__exit__.return_value = False
        patcher = patch.object(self.store, "connect", return_value=self.conn)
        self.connect = patcher.start()
        self.addCleanup(patcher.stop)
        self.columns = ["job_id", "profile_id", "namespace", "source_id", "chunk_id", "content_hash"]
        self.rows = [("job-a", "p", "code:one", "a.py", "a", "hash-a")]
        self.expiry = datetime(2026, 10, 30, tzinfo=timezone.utc)
        self.chunk = ("p", "code:one", "a.py", "a", "text", "hash-a", {"location": {"line": 1}}, "running")
        self.missing_job = self.missing_chunk = False
        self.calls = []
        self.conn.execute.side_effect = self.execute

    def execute(self, query, params):
        text = query.as_string()
        self.calls.append((text, params))
        result = MagicMock()
        if "SELECT j.job_id" in text:
            result.description = [SimpleNamespace(name=name) for name in self.columns]
            result.fetchall.return_value = self.rows
        elif "RETURNING attempts" in text:
            result.fetchone.return_value = None if self.missing_job else (2, params[0], self.expiry)
        else:
            result.fetchone.return_value = None if self.missing_chunk else self.chunk
        return result

    def test_scoped_current_version_locks_and_returned_lease(self):
        job = self.store.claim_jobs("p", 3, 60, 4)[0]
        text, params = self.calls[0]
        self.assertEqual(params, ("p", 4, 3))
        for clause in ("c.content_hash = j.content_hash", "FOR UPDATE OF j, c SKIP LOCKED",
                       "j.retry_after IS NULL", "j.lease_expires_at <= NOW()", "LIMIT %s"):
            self.assertIn(clause, text)
        self.assertEqual(job.id, "job-a")
        self.assertEqual(job.attempts, 2)
        self.assertEqual(job.lease_expires_at, self.expiry.timestamp())
        self.assertEqual(job.chunk.state, "running")
        self.assertEqual(job.lease_token, self.calls[1][1][0])
        self.assertEqual(self.calls[2][1], ("p", "code:one", "a.py", "a", "hash-a"))
        self.assertIn("retry_after = NULL, error_code = NULL", self.calls[1][0])
        job.chunk.metadata["location"]["line"] = 99
        self.assertEqual(self.chunk[6]["location"]["line"], 1)
        self.transaction.__exit__.assert_called_once_with(None, None, None)

    def test_no_candidates_produces_no_updates(self):
        self.rows = []
        self.assertEqual(self.store.claim_jobs("p", 3, 60, 4), [])
        self.assertEqual(len(self.calls), 1)

    def test_reclaim_uses_fresh_token(self):
        first = self.store.claim_jobs("p", 1, 60, 4)[0]
        second = self.store.claim_jobs("p", 1, 60, 4)[0]
        self.assertNotEqual(first.lease_token, second.lease_token)

    def test_invalid_arguments_never_connect(self):
        for args in (("", 1, 60, 4), ("p", 0, 60, 4), ("p", 1, -1, 4),
                     ("p", 1, 60, True), ("p", 1.5, 60, 4)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.store.claim_jobs(*args)
        self.connect.assert_not_called()

    def test_missing_job_or_chunk_aborts_transaction(self):
        for field in ("missing_job", "missing_chunk"):
            setattr(self, field, True)
            self.transaction.__exit__.reset_mock()
            with self.subTest(field=field), self.assertRaises(EmbeddingStorageError):
                self.store.claim_jobs("p", 1, 60, 4)
            self.assertIs(self.transaction.__exit__.call_args.args[0], EmbeddingStorageError)
            setattr(self, field, False)

    def test_failure_is_redacted_and_aborts_transaction(self):
        self.conn.execute.side_effect = RuntimeError("fake-test-password")
        with self.assertRaises(EmbeddingStorageError) as error:
            self.store.claim_jobs("p", 1, 60, 4)
        self.assertNotIn("fake-test-password", str(error.exception))
        self.assertIs(self.transaction.__exit__.call_args.args[0], RuntimeError)
