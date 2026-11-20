"""来源提交的输入边界与事务接线测试，不替代真实 PostgreSQL 事务验收。"""

from dataclasses import replace
import hashlib
import unittest
from unittest.mock import MagicMock, patch

from context_lab.embeddings import EmbeddingJob, IndexedChunk, PostgresEmbeddingStore
from context_lab.embeddings.errors import EmbeddingStorageError


class EmbeddingCommitTests(unittest.TestCase):
    def setUp(self):
        self.store = PostgresEmbeddingStore("postgresql://invalid.example/test", schema="test_commit")
        self.conn = MagicMock()
        self.conn.__enter__.return_value = self.conn
        self.conn.__exit__.return_value = False
        self.conn.transaction.return_value.__exit__.return_value = False
        self.saved_state = "pending"
        self.calls = []
        def execute(query, params=()):
            statement = query if isinstance(query, str) else query.as_string()
            self.calls.append((statement, params))
            cursor = MagicMock()
            cursor.fetchone.return_value = (self.saved_state,) if "RETURNING state" in statement else None
            return cursor
        self.conn.execute.side_effect = execute
        patcher = patch.object(self.store, "connect", return_value=self.conn)
        self.connect = patcher.start()
        self.addCleanup(patcher.stop)

    def chunk(self, chunk_id="get", text="read cache", state="pending", **changes):
        return IndexedChunk("p", "code:one", "a.py", chunk_id, text,
                            hashlib.sha256(text.encode()).hexdigest(), state=state, **changes)

    def test_valid_snapshot_saves_bound_values_without_sending_model_requests(self):
        chunk = self.chunk(metadata={"location": {"line": 1}})
        job = EmbeddingJob("job-a", chunk)
        self.assertIsNone(self.store.commit_source("p", "code:one", "a.py", [chunk], [job]))
        upserts = [(s, p) for s, p in self.calls if "RETURNING state" in s]
        self.assertEqual(len(upserts), 1)
        statement, params = upserts[0]
        self.assertIn('INSERT INTO "test_commit"."chunks" AS existing', statement)
        self.assertEqual(params[:6], ("p", "code:one", "a.py", "get", chunk.text, chunk.content_hash))
        self.assertEqual(params[-1], "pending")
        self.assertEqual(params[6].obj, chunk.metadata)
        self.assertIsNot(params[6].obj["location"], chunk.metadata["location"])
        inserts = [(s, p) for s, p in self.calls if 'INSERT INTO "test_commit"."jobs"' in s]
        self.assertEqual(inserts[0][1], ("job-a", "p", "code:one", "a.py", "get", chunk.content_hash))
        self.assertIn("DO NOTHING", inserts[0][0])
        self.conn.transaction.return_value.__exit__.assert_called_once_with(None, None, None)
        self.conn.__exit__.assert_called_once_with(None, None, None)

    def test_empty_snapshot_only_deletes_members_in_requested_source(self):
        self.store.commit_source("p", "code:one", "a.py", [], [])
        deletes = [(s, p) for s, p in self.calls if "DELETE FROM" in s]
        self.assertEqual(len(deletes), 1)
        self.assertIn('DELETE FROM "test_commit"."chunks"', deletes[0][0])
        self.assertEqual(deletes[0][1], ("p", "code:one", "a.py", []))
        self.assertFalse(any('"vectors"' in s for s, _ in self.calls))

    def test_unchanged_running_block_does_not_require_new_job_or_reset_lease(self):
        chunk = self.chunk(state="running")
        self.saved_state = "running"
        self.store.commit_source("p", "code:one", "a.py", [chunk], [])
        deletes = [(s, p) for s, p in self.calls if 'DELETE FROM "test_commit"."jobs"' in s]
        self.assertEqual(len(deletes), 1)
        self.assertIn("content_hash <> %s", deletes[0][0])
        self.assertEqual(deletes[0][1], ("p", "code:one", "a.py", "get", chunk.content_hash))
        self.assertFalse(any('INSERT INTO "test_commit"."jobs"' in s for s, _ in self.calls))
        self.assertFalse(any('UPDATE "test_commit"."jobs"' in s for s, _ in self.calls))

    def test_worker_advanced_ready_state_does_not_get_requeued_by_stale_plan(self):
        chunk = self.chunk()
        self.saved_state = "ready"
        self.store.commit_source("p", "code:one", "a.py", [chunk], [EmbeddingJob("job-a", chunk)])
        self.assertFalse(any('INSERT INTO "test_commit"."jobs"' in s for s, _ in self.calls))

    def test_missing_effective_task_aborts_transaction(self):
        original = self.conn.execute.side_effect
        def execute(query, params=()):
            result = original(query, params)
            statement = query if isinstance(query, str) else query.as_string()
            if "AND NOT EXISTS" in statement:
                result.fetchone.return_value = ("get",)
            return result
        self.conn.execute.side_effect = execute
        with self.assertRaisesRegex(EmbeddingStorageError, "缺少同版本有效任务"):
            self.store.commit_source("p", "code:one", "a.py", [self.chunk()], [])
        self.assertEqual(self.conn.transaction.return_value.__exit__.call_args.args[0], EmbeddingStorageError)

    def test_advisory_lock_is_stable_and_scoped(self):
        for source in ("a.py", "a.py", "b.py"):
            self.store.commit_source("p", "code:one", source, [], [])
        keys = [params[0] for statement, params in self.calls if "pg_advisory_xact_lock" in statement]
        self.assertEqual(keys[0], keys[1])
        self.assertNotEqual(keys[0], keys[2])

    def test_sql_failure_reaches_transaction_context_and_is_redacted(self):
        original = self.conn.execute.side_effect
        def execute(query, params=()):
            statement = query if isinstance(query, str) else query.as_string()
            if 'INSERT INTO "test_commit"."jobs"' in statement:
                raise RuntimeError("fake-test-password")
            return original(query, params)
        self.conn.execute.side_effect = execute
        chunk = self.chunk()
        with self.assertRaises(EmbeddingStorageError) as error:
            self.store.commit_source("p", "code:one", "a.py", [chunk], [EmbeddingJob("job-a", chunk)])
        self.assertNotIn("fake-test-password", str(error.exception))
        self.assertIn("RuntimeError", str(error.exception))
        self.assertEqual(self.conn.transaction.return_value.__exit__.call_args.args[0], RuntimeError)
        self.assertEqual(self.conn.__exit__.call_args.args[0], RuntimeError)

    def test_invalid_snapshots_are_rejected_before_database_access(self):
        chunk = self.chunk()
        bad_chunks = [
            None, [None], [replace(chunk, namespace="other")], [chunk, chunk],
            [replace(chunk, content_hash="wrong")], [replace(chunk, state="unknown")],
            [replace(chunk, metadata={"x": float("nan")})], [replace(chunk, metadata={"x": object()})],
        ]
        for chunks in bad_chunks:
            with self.subTest(chunks=type(chunks).__name__), self.assertRaises(ValueError):
                self.store.commit_source("p", "code:one", "a.py", chunks, [])
        self.connect.assert_not_called()

    def test_jobs_must_be_new_and_match_current_snapshot(self):
        chunk = self.chunk()
        job = EmbeddingJob("job-a", chunk)
        wrong_version = self.chunk(text="old body")
        cases = [
            None, [None], [replace(job, chunk=replace(chunk, source_id="other.py"))],
            [replace(job, chunk=wrong_version)], [replace(job, attempts=1)],
            [replace(job, lease_token="claimed")], [job, replace(job, id="job-b")],
            [replace(job, chunk=replace(chunk, chunk_id="absent"))],
        ]
        for jobs in cases:
            with self.subTest(jobs=type(jobs).__name__), self.assertRaises(ValueError):
                self.store.commit_source("p", "code:one", "a.py", [chunk], jobs)
        with self.assertRaises(ValueError):
            self.store.commit_source("p", "code:one", "a.py", [replace(chunk, state="ready")], [job])
        self.connect.assert_not_called()
