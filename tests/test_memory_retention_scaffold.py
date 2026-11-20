"""M08 外层装配与定时驱动；不执行真实清理算法。"""

from dataclasses import replace
from datetime import timedelta
from threading import Event
from unittest.mock import Mock, patch

from context_lab.memory import (
    MemoryCleanupPlan, MemoryCleanupReport, MemoryConfig, MemoryService, MemorySnapshot, MemorySourceRef,
)
from context_lab.tools.memory import MemoryTool
from tests.memory_support import NOW, record
from tests.support import SandboxTestCase


class MemoryRetentionScaffoldTests(SandboxTestCase):
    def service(self, enabled=False, clock=None):
        return MemoryService(self.tmp / "state" / "memory.json", "alice",
                             config=MemoryConfig(auto_cleanup_history=enabled), clock=clock or (lambda: NOW))

    def test_config_has_confirmed_retention_and_validates_schedule(self):
        config = MemoryConfig()
        self.assertEqual(config.history_retention_days, 90)
        self.assertFalse(config.auto_cleanup_history)
        self.assertEqual(config.history_cleanup_interval_seconds, 3600)
        self.assertEqual(config.history_cleanup_batch_size, 100)
        for name in ("history_retention_days", "history_cleanup_interval_seconds", "history_cleanup_batch_size"):
            for value in (0, -1, True, "90"):
                with self.subTest(field=name, value=value), self.assertRaises(ValueError):
                    replace(config, **{name: value})
        with self.assertRaises(ValueError):
            replace(config, auto_cleanup_history="true")

    def test_enabled_constructor_still_does_not_read_start_threads_or_call_clock(self):
        clock = Mock(side_effect=AssertionError("构造不能读时钟"))
        with patch("context_lab.storage.json_store.JsonFileStore.read", side_effect=AssertionError("构造不能读文件")):
            service = self.service(enabled=True, clock=clock)
        self.assertIs(service.retention.config, service.config)
        self.assertIs(service.history_cleanup_worker.service, service)
        self.assertIsNone(service.history_cleanup_worker.last_success_at)
        self.assertIsNone(service.last_cleanup_report)
        self.assertFalse(service.path.parent.exists())

    def test_plan_and_report_mutable_defaults_are_independent(self):
        cutoff = NOW - timedelta(days=90)
        a, b = MemoryCleanupPlan("alice", 0, NOW, cutoff), MemoryCleanupPlan("alice", 0, NOW, cutoff)
        a.candidates.append(MemorySourceRef("old", 2))
        a.protected["pinned"] = "pinned"
        self.assertEqual(b.candidates, [])
        self.assertEqual(b.protected, {})
        x, y = MemoryCleanupReport(), MemoryCleanupReport()
        x.deleted_ids.append("old")
        x.index_event_ids.append("delete-event")
        self.assertEqual(y.deleted_ids, [])
        self.assertEqual(y.index_event_ids, [])

    def test_preview_wrapper_binds_snapshot_user_clock_and_limit_without_writing(self):
        service = self.service()
        snapshot = MemorySnapshot(7, [record()], [])
        expected = MemoryCleanupPlan("alice", 7, NOW, NOW-timedelta(days=90))
        service.repository.read_snapshot = Mock(return_value=snapshot)
        service.repository.commit = Mock(side_effect=AssertionError("预览不能提交"))
        service.retention.plan = Mock(return_value=expected)
        self.assertIs(service.plan_history_cleanup(limit=5), expected)
        service.retention.plan.assert_called_once_with(snapshot, "alice", NOW, 5)
        service.repository.commit.assert_not_called()
        for limit in (-1, True):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                service.plan_history_cleanup(limit=limit)

    def test_tool_routes_cleanup_to_bound_service_and_rejects_user_override(self):
        service = self.service()
        service.plan_history_cleanup = Mock(return_value=MemoryCleanupPlan("alice", 0, NOW, NOW-timedelta(days=90)))
        service.cleanup_history = Mock(return_value=MemoryCleanupReport(dry_run=True))
        tool = MemoryTool(service.path, "alice", service=service)
        tool.run({"action": "plan_history_cleanup", "limit": 5})
        service.plan_history_cleanup.assert_called_once_with(limit=5)
        result = tool.run({"action": "cleanup_history", "dry_run": True, "limit": 5})
        self.assertTrue(result.success)
        service.cleanup_history.assert_called_once_with(dry_run=True, limit=5)
        with self.assertRaises(ValueError):
            tool.run({"action": "cleanup_history", "user_id": "bob"})

    def test_disabled_worker_does_not_read_or_attempt_cleanup(self):
        service = self.service(clock=Mock(side_effect=AssertionError("未启用不能取时间")))
        service.cleanup_history = Mock(side_effect=AssertionError("未启用不能清理"))
        self.assertIsNone(service.history_cleanup_worker.tick())
        self.assertFalse(service.path.parent.exists())

    def test_worker_only_runs_when_due_and_records_success(self):
        clock = Mock(side_effect=[NOW, NOW+timedelta(seconds=3599), NOW+timedelta(hours=1)])
        service = self.service(enabled=True, clock=clock)
        report = MemoryCleanupReport(dry_run=False)
        service.cleanup_history = Mock(return_value=report)
        worker = service.history_cleanup_worker
        self.assertIs(worker.tick(), report)
        self.assertIsNone(worker.tick())
        self.assertIs(worker.tick(), report)
        self.assertEqual(service.cleanup_history.call_count, 2)
        service.cleanup_history.assert_called_with(dry_run=False, limit=100)
        self.assertEqual(worker.last_success_at, NOW+timedelta(hours=1))
        self.assertIs(service.last_cleanup_report, report)

    def test_worker_errors_do_not_advance_schedule_or_report_false_success(self):
        service = self.service(enabled=True)
        report = MemoryCleanupReport(dry_run=False)
        service.cleanup_history = Mock(side_effect=[OSError("test unavailable"), report])
        worker = service.history_cleanup_worker
        with self.assertRaises(OSError):
            worker.tick()
        self.assertIsNone(worker.last_success_at)
        self.assertIsNone(service.last_cleanup_report)
        self.assertIs(worker.tick(), report)
        for bad_report in (None, MemoryCleanupReport(dry_run=True)):
            invalid = self.service(enabled=True)
            invalid.cleanup_history = Mock(return_value=bad_report)
            with self.subTest(report=bad_report), self.assertRaises(ValueError):
                invalid.history_cleanup_worker.tick()
            self.assertIsNone(invalid.history_cleanup_worker.last_success_at)

    def test_worker_loop_uses_interruptible_wait_and_stops_without_sleeping_in_tests(self):
        service = self.service(enabled=True)
        service.cleanup_history = Mock(return_value=MemoryCleanupReport(dry_run=False))

        class StopAfterWait:
            stopped = False

            def __init__(self):
                self.timeouts = []

            def is_set(self):
                return self.stopped

            def wait(self, timeout):
                self.timeouts.append(timeout)
                self.stopped = True

        stop = StopAfterWait()
        service.history_cleanup_worker.run(stop)
        self.assertEqual(stop.timeouts, [3600])
        service.cleanup_history.assert_called_once()
        already_stopped = Event()
        already_stopped.set()
        service.history_cleanup_worker.run(already_stopped)
        self.assertEqual(service.cleanup_history.call_count, 1)
