"""执行器调度测试，使用假客户端和模拟存储，不读取 .env。"""

import hashlib
import unittest
from unittest.mock import Mock, call

from context_lab.embeddings import (
    EmbeddingConfig, EmbeddingJob, HTTPEmbeddingClient, IncrementalEmbeddingIndex,
    IndexedChunk, PostgresEmbeddingStore, ProcessReport,
)
from context_lab.embeddings.errors import EmbeddingNotConfigured, EmbeddingRequestError, EmbeddingStorageError


class EmbeddingProcessTests(unittest.TestCase):
    def setUp(self):
        self.config = EmbeddingConfig(batch_size=2, lease_seconds=60, max_attempts=3)
        self.store = Mock(spec=PostgresEmbeddingStore)
        self.store.claim_jobs.return_value = []
        self.store.get_cached.return_value = None
        self.store.complete_job.return_value = True
        self.client = Mock()
        self.index = IncrementalEmbeddingIndex(self.config, client=self.client, store=self.store)

    def job(self, job_id, text, attempts=1):
        chunk = IndexedChunk(self.index.profile_id, "code:one", "a.py", job_id, text,
                             hashlib.sha256(text.encode()).hexdigest(), state="running")
        return EmbeddingJob(job_id, chunk, attempts, "lease-" + job_id, 123.0)

    def test_no_jobs_returns_zero_without_model_call(self):
        self.assertEqual(self.index.process_pending(), 0)
        self.store.claim_jobs.assert_called_once_with(self.index.profile_id, 2, 60, 3)
        self.client.embed.assert_not_called()
        self.assertEqual(self.index.last_process_report, ProcessReport())

    def test_same_hash_embeds_once_and_submits_each_job(self):
        first, second, third = self.job("a", "same"), self.job("b", "same"), self.job("c", "other")
        self.store.claim_jobs.return_value = [first, second, third]
        self.client.embed.return_value = [[1.0, 1.0], [2.0, 1.0]]
        self.assertEqual(self.index.process_pending(3), 3)
        self.client.embed.assert_called_once_with(["same", "other"])
        self.assertEqual(self.store.complete_job.call_args_list, [
            call(first, [1.0, 1.0]), call(second, [1.0, 1.0]), call(third, [2.0, 1.0]),
        ])
        self.assertEqual(self.index.last_process_report, ProcessReport(claimed=3, generated=2, completed=3))

    def test_all_cache_hits_skip_model_and_count_tasks(self):
        self.store.claim_jobs.return_value = [self.job("a", "same"), self.job("b", "same")]
        self.store.get_cached.return_value = [1.0, 2.0]
        self.assertEqual(self.index.process_pending(2), 2)
        self.client.embed.assert_not_called()
        self.store.get_cached.assert_called_once()
        self.assertEqual(self.index.last_process_report, ProcessReport(claimed=2, cache_hits=2, completed=2))

    def test_cached_tasks_complete_before_missing_batch_failure(self):
        cached, missing = self.job("a", "cached"), self.job("b", "missing")
        self.store.claim_jobs.return_value = [cached, missing]
        self.store.get_cached.side_effect = lambda profile, digest: [1.0, 1.0] if digest == cached.chunk.content_hash else None
        def fail(texts):
            self.store.complete_job.assert_called_once_with(cached, [1.0, 1.0])
            raise EmbeddingRequestError("timeout", "timeout", True)
        self.client.embed.side_effect = fail
        self.assertEqual(self.index.process_pending(), 1)
        self.store.fail_job.assert_called_once_with(missing, error_code="timeout", retryable=True)
        self.assertEqual(self.index.last_process_report.generation_errors, 1)

    def test_failed_batch_does_not_discard_prior_success_or_stop_next_batch(self):
        jobs = [self.job(str(i), str(i)) for i in range(5)]
        self.store.claim_jobs.return_value = jobs
        self.client.embed.side_effect = [
            [[1.0, 1.0], [2.0, 1.0]],
            EmbeddingRequestError("busy", "http_503", True),
            [[5.0, 1.0]],
        ]
        self.assertEqual(self.index.process_pending(5), 3)
        self.assertEqual(self.client.embed.call_args_list, [call(["0", "1"]), call(["2", "3"]), call(["4"])])
        self.assertEqual([item.args[0] for item in self.store.complete_job.call_args_list], [jobs[0], jobs[1], jobs[4]])
        self.assertEqual([item.args[0] for item in self.store.fail_job.call_args_list], jobs[2:4])
        self.assertEqual(self.index.last_process_report,
                         ProcessReport(claimed=5, generated=3, completed=3, generation_errors=2, request_errors=1))

    def test_retry_decision_uses_error_category_and_actual_attempts(self):
        first, last = self.job("a", "same", 1), self.job("b", "same", 3)
        self.store.claim_jobs.return_value = [first, last]
        self.client.embed.side_effect = EmbeddingRequestError("timeout", "timeout", True)
        self.assertEqual(self.index.process_pending(), 0)
        self.assertEqual(self.store.fail_job.call_args_list, [
            call(first, error_code="timeout", retryable=True),
            call(last, error_code="timeout", retryable=False),
        ])
        self.assertEqual(self.index.last_process_report.generation_errors, 2)
        self.assertEqual(self.index.last_process_report.request_errors, 1)

    def test_nonretryable_request_error_does_not_retry(self):
        job = self.job("a", "same")
        self.store.claim_jobs.return_value = [job]
        self.client.embed.side_effect = EmbeddingRequestError("auth", "http_401", False)
        self.assertEqual(self.index.process_pending(), 0)
        self.store.fail_job.assert_called_once_with(job, error_code="http_401", retryable=False)

    def test_stale_submission_is_not_counted_or_reported_as_failure(self):
        self.store.claim_jobs.return_value = [self.job("a", "same")]
        self.store.get_cached.return_value = [1.0, 1.0]
        self.store.complete_job.return_value = False
        self.assertEqual(self.index.process_pending(), 0)
        self.assertEqual(self.index.last_process_report.stale, 1)
        self.store.fail_job.assert_not_called()

    def test_wrong_response_count_does_not_silently_drop_jobs(self):
        self.store.claim_jobs.return_value = [self.job("a", "one"), self.job("b", "two")]
        self.client.embed.return_value = [[1.0, 1.0]]
        self.assertEqual(self.index.process_pending(), 0)
        self.store.complete_job.assert_not_called()
        self.assertEqual(self.store.fail_job.call_count, 2)
        self.assertTrue(all(item.kwargs == {"error_code": "invalid_response", "retryable": False}
                            for item in self.store.fail_job.call_args_list))

    def test_storage_failure_is_propagated_not_reported_as_model_failure(self):
        self.store.claim_jobs.return_value = [self.job("a", "same")]
        self.store.get_cached.return_value = [1.0, 1.0]
        self.store.complete_job.side_effect = EmbeddingStorageError("test database failure")
        with self.assertRaises(EmbeddingStorageError):
            self.index.process_pending()
        self.store.fail_job.assert_not_called()
        self.client.embed.assert_not_called()

    def test_failure_report_storage_error_is_not_swallowed(self):
        self.store.claim_jobs.return_value = [self.job("a", "same")]
        self.client.embed.side_effect = EmbeddingRequestError("timeout", "timeout", True)
        self.store.fail_job.side_effect = EmbeddingStorageError("test database failure")
        with self.assertRaises(EmbeddingStorageError):
            self.index.process_pending()

    def test_unconfigured_http_client_stops_before_claim(self):
        index = IncrementalEmbeddingIndex(self.config, client=HTTPEmbeddingClient(self.config), store=self.store)
        with self.assertRaises(EmbeddingNotConfigured):
            index.process_pending()
        self.store.claim_jobs.assert_not_called()

    def test_bad_limits_never_claim(self):
        for limit in (0, -1, True, 1.5, "2"):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                self.index.process_pending(limit)
        self.store.claim_jobs.assert_not_called()

    def test_report_is_reset_per_run(self):
        self.store.claim_jobs.return_value = [self.job("a", "same")]
        self.store.get_cached.return_value = [1.0, 1.0]
        self.assertEqual(self.index.process_pending(), 1)
        previous = self.index.last_process_report
        self.store.claim_jobs.return_value = []
        self.assertEqual(self.index.process_pending(), 0)
        self.assertEqual(previous.completed, 1)
        self.assertEqual(self.index.last_process_report, ProcessReport())
