"""真实事务验收：显式设置 TEST_DATABASE_URL 后单独运行，不使用业务库或跳过测试。"""

from datetime import datetime, timedelta, timezone
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from threading import Barrier
import unittest
from unittest.mock import Mock
from uuid import uuid4

import psycopg
from psycopg import sql
from pgvector import Vector

from context_lab.embeddings import (
    EmbeddingChunk, EmbeddingConfig, EmbeddingJob, IncrementalEmbeddingIndex,
    IndexedChunk, PostgresEmbeddingStore,
)
from context_lab.embeddings.errors import EmbeddingRequestError, EmbeddingStorageError, IndexNotReady


class EmbeddingCommitPostgresTests(unittest.TestCase):
    def setUp(self):
        url = os.environ.get("TEST_DATABASE_URL", "")
        if not url.strip():
            raise RuntimeError("真实事务验收需要显式 TEST_DATABASE_URL；不会使用 DATABASE_URL")
        self.store = PostgresEmbeddingStore(url, schema="test_commit_" + uuid4().hex)
        self.store.initialize()
        self.addCleanup(self.drop_schema)

    def drop_schema(self):
        if not self.store.schema.startswith("test_commit_"):
            raise RuntimeError("拒绝清理非测试 schema")
        with psycopg.connect(self.store.database_url) as conn:
            conn.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(self.store.schema)))

    @staticmethod
    def chunk(chunk_id, text, source="a.py", state="pending"):
        return IndexedChunk("p", "code:one", source, chunk_id, text,
                            hashlib.sha256(text.encode()).hexdigest(), state=state)

    def jobs_for(self, chunks, prefix):
        return [EmbeddingJob(prefix + chunk.chunk_id, chunk) for chunk in chunks if chunk.state == "pending"]

    def tasks(self):
        with self.store.connect() as conn:
            return conn.execute(sql.SQL("""
                SELECT job_id, source_id, chunk_id, content_hash, state, attempts, lease_token
                FROM {} ORDER BY job_id
            """).format(sql.Identifier(self.store.schema, "jobs"))).fetchall()

    def add_cache(self, chunk):
        with self.store.connect() as conn:
            conn.execute(sql.SQL("INSERT INTO {} (profile_id, content_hash, embedding) VALUES (%s, %s, %s)")
                         .format(sql.Identifier(self.store.schema, "vectors")),
                         ("p", chunk.content_hash, Vector([1.0, 2.0])))

    def test_change_delete_and_cache_hit_preserve_unrelated_and_running_tasks(self):
        original = [self.chunk("a", "alpha"), self.chunk("b", "beta"), self.chunk("c", "gamma")]
        self.store.commit_source("p", "code:one", "a.py", original, self.jobs_for(original, "old-"))
        other = [self.chunk("x", "other", source="b.py")]
        self.store.commit_source("p", "code:one", "b.py", other, self.jobs_for(other, "other-"))
        cached = self.chunk("d", "shared", state="ready")
        self.add_cache(cached)
        self.add_cache(original[0])
        with self.store.connect() as conn:
            conn.execute(sql.SQL("""
                UPDATE {} SET state='running'
                WHERE profile_id='p' AND namespace='code:one' AND source_id='a.py' AND chunk_id='b'
            """).format(sql.Identifier(self.store.schema, "chunks")))
            conn.execute(sql.SQL("""
                UPDATE {} SET state='running', attempts=1, lease_token='lease-b', lease_expires_at=%s
                WHERE job_id='old-b'
            """).format(sql.Identifier(self.store.schema, "jobs")),
                         (datetime.now(timezone.utc) + timedelta(hours=1),))
        changed = self.chunk("a", "alpha modified")
        # b 的调用方快照仍为旧 pending，数据库实际已 running，不能倒退。
        self.store.commit_source("p", "code:one", "a.py", [changed, original[1], cached], [EmbeddingJob("new-a", changed)])
        restored = {chunk.chunk_id: chunk for chunk in self.store.list_source("p", "code:one", "a.py")}
        self.assertEqual(set(restored), {"a", "b", "d"})
        self.assertEqual(restored["a"].text, "alpha modified")
        self.assertEqual(restored["b"].state, "running")
        self.assertEqual(restored["d"].state, "ready")
        tasks = {row[0]: row for row in self.tasks()}
        self.assertEqual(set(tasks), {"new-a", "old-b", "other-x"})
        self.assertEqual(tasks["old-b"][4:], ("running", 1, "lease-b"))
        self.assertEqual(self.store.get_cached("p", original[0].content_hash), [1.0, 2.0])
        self.assertEqual(len(self.store.list_source("p", "code:one", "b.py")), 1)

    def test_duplicate_task_and_empty_snapshot_do_not_reset_or_cross_sources(self):
        chunk = self.chunk("a", "alpha")
        self.store.commit_source("p", "code:one", "a.py", [chunk], [EmbeddingJob("first-a", chunk)])
        self.store.commit_source("p", "code:one", "a.py", [chunk], [EmbeddingJob("second-a", chunk)])
        self.assertEqual([row[0] for row in self.tasks()], ["first-a"])
        self.add_cache(chunk)
        other = self.chunk("b", "other", source="b.py")
        self.store.commit_source("p", "code:one", "b.py", [other], [EmbeddingJob("other-b", other)])
        self.store.commit_source("p", "code:one", "a.py", [], [])
        self.store.commit_source("p", "code:one", "a.py", [], [])
        self.assertEqual(self.store.list_source("p", "code:one", "a.py"), [])
        self.assertEqual([row[0] for row in self.tasks()], ["other-b"])
        self.assertEqual(self.store.get_cached("p", chunk.content_hash), [1.0, 2.0])

    def test_late_database_failure_rolls_back_earlier_chunk_and_job_changes(self):
        chunk = self.chunk("a", "alpha")
        self.store.commit_source("p", "code:one", "a.py", [chunk], [EmbeddingJob("old-a", chunk)])
        other = self.chunk("b", "other", source="b.py")
        self.store.commit_source("p", "code:one", "b.py", [other], [EmbeddingJob("collision", other)])
        before = self.tasks()
        changed = self.chunk("a", "new text")
        # 与另一来源的 job_id 冲突；块修改和旧任务删除必须一起回滚。
        with self.assertRaises(EmbeddingStorageError):
            self.store.commit_source("p", "code:one", "a.py", [changed], [EmbeddingJob("collision", changed)])
        self.assertEqual(self.store.list_source("p", "code:one", "a.py")[0].text, "alpha")
        self.assertEqual(self.tasks(), before)

    def test_namespace_claim_leaves_other_namespaces_pending(self):
        target = self.chunk("target", "current")
        other = replace(self.chunk("other", "unrelated"), namespace="code:two")
        self.store.commit_source("p", other.namespace, "a.py", [other], [EmbeddingJob("a-other", other)])
        self.store.commit_source("p", target.namespace, "a.py", [target], [EmbeddingJob("b-target", target)])

        claimed = self.store.claim_jobs("p", 2, 60, 3, namespace=target.namespace)

        self.assertEqual([job.id for job in claimed], ["b-target"])
        self.assertEqual(self.store.status("p", other.namespace).pending, 1)

    def test_vector_repair_cli_uses_memory_and_frozen_source(self):
        from contextlib import redirect_stdout
        from io import StringIO
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch

        from context_lab.agents import CodebaseMaintainer
        from context_lab.cli import main
        from context_lab.memory import MemoryService
        from context_lab.tools.sandbox import DockerSandbox

        with TemporaryDirectory() as folder:
            state_dir = Path(folder)
            report = state_dir / "result.json"
            memory = "normalize_query must preserve interior spaces"
            MemoryService(state_dir / "memories.json", "student").add(memory)
            env = {
                "EMBEDDING_BASE_URL": "https://embedding.example/v1",
                "EMBEDDING_MODEL": "test-embedding", "EMBEDDING_API_KEY": "test-key",
                "EMBEDDING_EXPECTED_DIMENSIONS": "2",
                "EMBEDDING_DB_SCHEMA": self.store.schema, "DATABASE_URL": self.store.database_url,
            }
            agents = []

            def make_agent(*args, **kwargs):
                agent = CodebaseMaintainer(*args, **kwargs)
                agents.append(agent)
                return agent

            def sandbox_test(sandbox, candidate, command, tests):
                source = (candidate / "src/search.py").read_text(encoding="utf-8")
                passes = "test_plain_query" in command[-1] or "query.strip().lower()" in source
                return {"status": "passed" if passes else "failed", "exit_code": int(not passes), "output": "tests"}

            with (
                patch.dict(os.environ, env),
                patch("context_lab.cli.CodebaseMaintainer", side_effect=make_agent),
                patch("context_lab.embeddings.HTTPEmbeddingClient.embed",
                      side_effect=lambda texts: [[1.0, 2.0] for _ in texts]) as encode,
                patch.object(DockerSandbox, "ensure_ready"),
                patch.object(DockerSandbox, "_run_test", autospec=True, side_effect=sandbox_test),
                redirect_stdout(StringIO()),
            ):
                exit_code = main([
                    "repair", "--sample", "--retrieval", "vector", "--state-dir", folder,
                    "--issue", "normalize_query", "--report-out", str(report),
                ])

            self.assertEqual(exit_code, 0)
            result = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(result["result"]["status"], "passed")
            stages = {stage["stage"]: stage for stage in result["retrieval"]}
            self.assertEqual(stages["memory_index"]["ready"], 1)
            self.assertEqual(stages["rag_index"]["mode"], "vector")
            self.assertGreater(stages["rag_index"]["ready"], 0)
            self.assertIn(memory, agents[0].model.calls[0][-1].content)
            self.assertIn("src/search.py:", agents[0].model.calls[1][-1].content)
            self.assertEqual(encode.call_args_list[0].args[0], [memory])
            self.assertEqual(encode.call_args_list[-1].args[0], ["normalize_query"])

    def test_claim_then_complete_saves_dimension_cache_and_clears_lease(self):
        chunk = self.chunk("a", "alpha")
        self.store.commit_source("p", "code:one", "a.py", [chunk], [EmbeddingJob("job-a", chunk)])
        claimed = self.store.claim_jobs("p", 1, 60, 3)[0]
        self.assertTrue(self.store.complete_job(claimed, [1.0, 2.0]))
        self.assertEqual(self.store.list_source("p", "code:one", "a.py")[0].state, "ready")
        self.assertEqual(self.store.get_cached("p", chunk.content_hash), [1.0, 2.0])
        with self.store.connect() as conn:
            profile = conn.execute(sql.SQL("SELECT dimensions FROM {} WHERE profile_id='p'")
                                   .format(sql.Identifier(self.store.schema, "profiles"))).fetchone()
            task = conn.execute(sql.SQL("""
                SELECT state, attempts, lease_token, lease_expires_at, retry_after, error_code
                FROM {} WHERE job_id='job-a'
            """).format(sql.Identifier(self.store.schema, "jobs"))).fetchone()
        self.assertEqual(profile, (2,))
        self.assertEqual(task, ("ready", 1, None, None, None, None))
        self.assertFalse(self.store.complete_job(claimed, [1.0, 2.0]))

    def test_completion_rejects_wrong_expired_changed_and_deleted_jobs(self):
        chunk = self.chunk("a", "alpha")
        self.store.commit_source("p", "code:one", "a.py", [chunk], [EmbeddingJob("job-a", chunk)])
        claimed = self.store.claim_jobs("p", 1, 60, 3)[0]
        self.assertFalse(self.store.complete_job(replace(claimed, lease_token="wrong"), [1.0, 2.0]))
        with self.store.connect() as conn:
            conn.execute(sql.SQL("UPDATE {} SET lease_expires_at=clock_timestamp()-INTERVAL '1 second'")
                         .format(sql.Identifier(self.store.schema, "jobs")))
        self.assertFalse(self.store.complete_job(claimed, [1.0, 2.0]))
        reclaimed = self.store.claim_jobs("p", 1, 60, 3)[0]
        self.assertNotEqual(claimed.lease_token, reclaimed.lease_token)
        self.assertFalse(self.store.complete_job(claimed, [1.0, 2.0]))
        changed = self.chunk("a", "new text")
        self.store.commit_source("p", "code:one", "a.py", [changed], [EmbeddingJob("job-new", changed)])
        self.assertFalse(self.store.complete_job(reclaimed, [1.0, 2.0]))
        latest = self.store.claim_jobs("p", 1, 60, 3)[0]
        self.store.commit_source("p", "code:one", "a.py", [], [])
        self.assertFalse(self.store.complete_job(latest, [1.0, 2.0]))
        self.assertIsNone(self.store.get_cached("p", chunk.content_hash))
        self.assertIsNone(self.store.get_cached("p", changed.content_hash))

    def test_completion_late_constraint_failure_rolls_back_all_four_tables(self):
        chunk = self.chunk("a", "alpha")
        self.store.commit_source("p", "code:one", "a.py", [chunk], [EmbeddingJob("job-a", chunk)])
        claimed = self.store.claim_jobs("p", 1, 60, 3)[0]
        before = self.tasks()
        with self.store.connect() as conn:
            conn.execute(sql.SQL("ALTER TABLE {} ADD CHECK (state <> 'ready')")
                         .format(sql.Identifier(self.store.schema, "chunks")))
        with self.assertRaises(EmbeddingStorageError):
            self.store.complete_job(claimed, [1.0, 2.0])
        self.assertEqual(self.tasks(), before)
        self.assertEqual(self.store.list_source("p", "code:one", "a.py")[0].state, "running")
        self.assertIsNone(self.store.get_cached("p", chunk.content_hash))
        with self.store.connect() as conn:
            profile = conn.execute(sql.SQL("SELECT dimensions FROM {} WHERE profile_id='p'")
                                   .format(sql.Identifier(self.store.schema, "profiles"))).fetchone()
        self.assertEqual(profile, (None,))

    def test_failure_retry_and_terminal_state_preserve_other_tasks_and_cache(self):
        chunks = [self.chunk("a", "alpha"), self.chunk("b", "beta")]
        self.store.commit_source("p", "code:one", "a.py", chunks, self.jobs_for(chunks, "job-"))
        claimed = {job.id: job for job in self.store.claim_jobs("p", 2, 60, 3)}
        self.add_cache(chunks[0])
        before_b = next(row for row in self.tasks() if row[0] == "job-b")
        self.store.fail_job(claimed["job-a"], "timeout", True)
        self.assertEqual(next(row for row in self.tasks() if row[0] == "job-b"), before_b)
        self.store.fail_job(claimed["job-b"], "bad_model", False)
        with self.store.connect() as conn:
            tasks = conn.execute(sql.SQL("""
                SELECT job_id, state, attempts, lease_token, lease_expires_at,
                       retry_after IS NULL, retry_after > clock_timestamp(), error_code
                FROM {} ORDER BY job_id
            """).format(sql.Identifier(self.store.schema, "jobs"))).fetchall()
        self.assertEqual(tasks, [
            ("job-a", "pending", 1, None, None, False, True, "timeout"),
            ("job-b", "failed", 1, None, None, True, None, "bad_model"),
        ])
        states = {chunk.chunk_id: chunk.state for chunk in self.store.list_source("p", "code:one", "a.py")}
        self.assertEqual(states, {"a": "pending", "b": "failed"})
        self.assertEqual(self.store.get_cached("p", chunks[0].content_hash), [1.0, 2.0])
        before = self.tasks()
        self.store.fail_job(claimed["job-a"], "old_report", False)
        self.assertEqual(self.tasks(), before)
        self.assertEqual(self.store.claim_jobs("p", 2, 60, 3), [])

    def test_failure_late_chunk_constraint_rolls_back_task_update(self):
        chunk = self.chunk("a", "alpha")
        self.store.commit_source("p", "code:one", "a.py", [chunk], [EmbeddingJob("job-a", chunk)])
        claimed = self.store.claim_jobs("p", 1, 60, 3)[0]
        before = self.tasks()
        with self.store.connect() as conn:
            conn.execute(sql.SQL("ALTER TABLE {} ADD CHECK (state <> 'pending')")
                         .format(sql.Identifier(self.store.schema, "chunks")))
        with self.assertRaises(EmbeddingStorageError):
            self.store.fail_job(claimed, "timeout", True)
        self.assertEqual(self.tasks(), before)
        self.assertEqual(self.store.list_source("p", "code:one", "a.py")[0].state, "running")

    def test_process_pipeline_deduplicates_and_reuses_persistent_cache(self):
        config = EmbeddingConfig(database_url=self.store.database_url, database_schema=self.store.schema,
                                 batch_size=2, expected_dimensions=2)
        client = Mock()
        client.embed.side_effect = lambda texts: [[float(len(text)), 1.0] for text in texts]
        index = IncrementalEmbeddingIndex(config, client=client, store=self.store)
        index.sync_source("code:worker", "a.py", [
            EmbeddingChunk("a", "same"), EmbeddingChunk("b", "same"), EmbeddingChunk("c", "other"),
        ])
        self.assertEqual(index.process_pending(3), 3)
        client.embed.assert_called_once()
        self.assertCountEqual(client.embed.call_args.args[0], ["same", "other"])
        self.assertEqual(index.last_process_report.generated, 2)
        self.assertEqual(index.status("code:worker").ready, 3)
        # 新实例、新来源复用数据库缓存，无需再次调用模型。
        restored = IncrementalEmbeddingIndex(config, client=client)
        report = restored.sync_source("code:worker", "b.py", [EmbeddingChunk("d", "same")])
        self.assertEqual(report.cache_hits, 1)
        self.assertEqual(restored.process_pending(), 0)
        self.assertEqual(client.embed.call_count, 1)
        self.assertEqual(restored.status("code:worker").ready, 4)
        ready = self.store.list_ready(restored.profile_id, "code:worker")
        self.assertEqual(len(ready), 4)
        self.assertEqual({chunk.chunk_id for chunk, _ in ready}, {"a", "b", "c", "d"})
        self.assertTrue(all(vector == [float(len(chunk.text)), 1.0] for chunk, vector in ready))
        self.assertEqual(len(self.store.list_ready(restored.profile_id, "code:worker", "a.py")), 3)
        self.assertEqual(self.store.list_ready(restored.profile_id, "code:unrelated"), [])
        hits = self.store.search_ready(restored.profile_id, "code:worker", [4.0, 1.0], limit=2)
        self.assertEqual([hit.chunk.chunk_id for hit in hits], ["a", "b"])
        self.assertTrue(all(abs(hit.score - 1.0) < 1e-6 for hit in hits))
        source_hits = self.store.search_ready(restored.profile_id, "code:worker", [4.0, 1.0], source_id="b.py")
        self.assertEqual([hit.chunk.chunk_id for hit in source_hits], ["d"])
        self.assertEqual(self.store.search_ready(restored.profile_id, "code:unrelated", [4.0, 1.0]), [])
        with self.assertRaises(EmbeddingStorageError):
            self.store.search_ready(restored.profile_id, "code:worker", [1.0, 2.0, 3.0])
        query_hits = restored.search("same", "code:worker", limit=2)
        self.assertEqual([hit.chunk.chunk_id for hit in query_hits], ["a", "b"])
        self.assertEqual(client.embed.call_count, 2)
        # c 换成新正文但尚未生成向量，旧缓存不能作为其当前结果返回。
        restored.sync_source("code:worker", "a.py", [
            EmbeddingChunk("a", "same"), EmbeddingChunk("b", "same"), EmbeddingChunk("c", "changed"),
        ])
        ready = self.store.list_ready(restored.profile_id, "code:worker")
        self.assertEqual({chunk.chunk_id for chunk, _ in ready}, {"a", "b", "d"})
        with self.assertRaises(IndexNotReady):
            restored.search("same", "code:worker")
        self.assertEqual(client.embed.call_count, 2)

    def test_process_pipeline_stops_retrying_after_attempt_limit(self):
        config = EmbeddingConfig(database_url=self.store.database_url, database_schema=self.store.schema,
                                 batch_size=2, max_attempts=2, expected_dimensions=2)
        client = Mock()
        client.embed.side_effect = EmbeddingRequestError("timeout", "timeout", True)
        index = IncrementalEmbeddingIndex(config, client=client, store=self.store)
        index.sync_source("code:worker", "a.py", [EmbeddingChunk("a", "text")])
        self.assertEqual(index.process_pending(), 0)
        self.assertEqual(index.status("code:worker").pending, 1)
        with self.store.connect() as conn:
            conn.execute(sql.SQL("UPDATE {} SET retry_after=clock_timestamp()-INTERVAL '1 second'")
                         .format(sql.Identifier(self.store.schema, "jobs")))
        self.assertEqual(index.process_pending(), 0)
        self.assertEqual(index.status("code:worker").failed, 1)
        self.assertEqual(index.process_pending(), 0)
        self.assertEqual(client.embed.call_count, 2)
        with self.store.connect() as conn:
            task = conn.execute(sql.SQL("SELECT state, attempts, error_code FROM {}")
                                .format(sql.Identifier(self.store.schema, "jobs"))).fetchone()
        self.assertEqual(task, ("failed", 2, "timeout"))

    def test_parallel_workers_claim_distinct_jobs(self):
        chunks = [self.chunk(str(i), f"text {i}") for i in range(6)]
        self.store.commit_source("p", "code:one", "a.py", chunks, self.jobs_for(chunks, "job-"))
        barrier = Barrier(2)
        def claim():
            barrier.wait(timeout=5)
            return self.store.claim_jobs("p", 3, 60, 3)
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(claim) for _ in range(2)]
            batches = [future.result(timeout=10) for future in futures]
        ids = [job.id for batch in batches for job in batch]
        self.assertEqual(len(ids), 6)
        self.assertEqual(len(set(ids)), 6)
        self.assertTrue(all(job.attempts == 1 and job.lease_token for batch in batches for job in batch))

    def test_claim_skips_task_locked_by_another_connection(self):
        chunks = [self.chunk("a", "alpha"), self.chunk("b", "beta")]
        self.store.commit_source("p", "code:one", "a.py", chunks, self.jobs_for(chunks, "job-"))
        with self.store.connect() as holder:
            holder.execute(sql.SQL("SELECT job_id FROM {} WHERE job_id='job-a' FOR UPDATE")
                           .format(sql.Identifier(self.store.schema, "jobs")))
            # 若缺少 SKIP LOCKED，则这次领取会因锁等待超时而失败。
            limited = PostgresEmbeddingStore(
                self.store.database_url, schema=self.store.schema,
                connection_factory=lambda url, **kwargs: psycopg.connect(url, options="-c lock_timeout=2000", **kwargs),
            )
            claimed = limited.claim_jobs("p", 2, 60, 3)
            self.assertEqual([job.id for job in claimed], ["job-b"])
        claimed = self.store.claim_jobs("p", 2, 60, 3)
        self.assertEqual([job.id for job in claimed], ["job-a"])
