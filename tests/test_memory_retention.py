"""M08：保留策略、批量清理、并发重规划与维护驱动的离线验收。"""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from context_lab.memory import (
    MemoryConfig, MemoryConflict, MemoryIndexEvent, MemoryRetentionPolicy, MemoryService,
    MemorySnapshot, MemorySourceRef,
)
from tests.memory_support import NOW, record
from tests.support import SandboxTestCase


def expired(memory_id="expired", days=91, **changes):
    return record(memory_id, fact_key=f"project.{memory_id}.database", valid_from=NOW-timedelta(days=200),
                  valid_to=NOW-timedelta(days=days), **changes)


def ordinary(memory_id="ordinary", days=91, **changes):
    return record(memory_id, **{
        "created_at": NOW-timedelta(days=200), "updated_at": NOW-timedelta(days=days),
        **changes,
    })


class MemoryRetentionTests(SandboxTestCase):
    def service(self, user_id="alice"):
        return MemoryService(self.tmp / "memory.json", user_id,
                             config=MemoryConfig(history_retention_days=90), clock=lambda: NOW)

    def seed(self):
        service = self.service()
        old = expired()
        current = record("current", fact_key="project.current.database", supersedes_id=old.id)
        pinned = expired("pinned", metadata={"pinned": True})
        event = MemoryIndexEvent("old-upsert", "alice", old.id, "upsert", old.version, NOW)
        service.repository.commit(MemorySnapshot(0, [old, current, pinned], [event]))
        return service, old, current, pinned

    def test_retention_uses_valid_to_including_boundary_not_recent_update_time(self):
        old = expired(updated_at=NOW)
        boundary = expired("boundary", days=90)
        recent = expired("recent", days=89)
        snapshot = MemorySnapshot(3, [old, boundary, recent])
        plan = MemoryRetentionPolicy().plan(snapshot, "alice", NOW)
        self.assertEqual(plan.cutoff, NOW-timedelta(days=90))
        self.assertEqual(plan.revision, 3)
        self.assertEqual([ref.memory_id for ref in plan.candidates], [old.id, boundary.id])

    def test_only_eligible_archives_are_candidates_and_pins_and_fresh_evidence_are_protected(self):
        pinned = expired("pinned", metadata={"pinned": True})
        cited, uncited = expired("cited"), expired("uncited")
        active = record("active", fact_key="project.active.database")
        future = record("future", fact_key="project.future.database", valid_from=NOW+timedelta(days=1))
        ordinary = record("ordinary", valid_to=NOW-timedelta(days=100))
        working = expired("working", layer="working", task_id="task-1")
        summary = record("summary", layer="summary", source_refs=[MemorySourceRef(cited.id, cited.version)])
        stale = record("stale", layer="summary", summary_state="stale",
                       source_refs=[MemorySourceRef(uncited.id, uncited.version)])
        foreign = expired(uncited.id, user_id="bob")
        foreign_summary = record("bob-summary", user_id="bob", layer="summary",
                                 source_refs=[MemorySourceRef(uncited.id, uncited.version)])
        plan = MemoryRetentionPolicy().plan(MemorySnapshot(0, [pinned, cited, uncited, active, future,
                                                                  ordinary, working, summary, stale, foreign, foreign_summary]),
                                             "alice", NOW)
        self.assertEqual(plan.candidates, [MemorySourceRef(ordinary.id, ordinary.version),
                                          MemorySourceRef(uncited.id, uncited.version)])
        self.assertEqual(plan.protected, {pinned.id: "pinned", cited.id: "summary_reference"})

    def test_plan_is_deterministic_batched_and_does_not_modify_input(self):
        a, b, c = expired("a", days=100), expired("b", days=100), expired("c", days=91)
        snapshot = MemorySnapshot(7, [c, b, a])
        before = deepcopy(snapshot)
        policy = MemoryRetentionPolicy(MemoryConfig(history_cleanup_batch_size=2))
        self.assertEqual([ref.memory_id for ref in policy.plan(snapshot, "alice", NOW).candidates], ["a", "b"])
        self.assertEqual([ref.memory_id for ref in policy.plan(snapshot, "alice", NOW, limit=1).candidates], ["a"])
        self.assertEqual(policy.plan(snapshot, "alice", NOW, limit=0).candidates, [])
        self.assertEqual(snapshot, before)

    def test_ordinary_archive_uses_last_update_not_creation_time_and_includes_boundary(self):
        old = ordinary("old", days=100)
        boundary = ordinary("boundary", days=90)
        recently_updated = ordinary("recent", days=1)
        active_fact = record("active-fact", fact_key="project.database",
                             created_at=NOW-timedelta(days=200), updated_at=NOW-timedelta(days=100))
        snapshot = MemorySnapshot(4, [recently_updated, active_fact, boundary, old])
        before = deepcopy(snapshot)
        plan = MemoryRetentionPolicy().plan(snapshot, "alice", NOW)
        self.assertEqual(plan.candidates, [MemorySourceRef(old.id, old.version),
                                          MemorySourceRef(boundary.id, boundary.version)])
        self.assertEqual(snapshot, before)

    def test_ordinary_archive_with_explicit_validity_uses_valid_to_and_keeps_future_records(self):
        old_end = ordinary("old-end", days=0, valid_to=NOW-timedelta(days=100))
        recent_end = ordinary("recent-end", days=120, valid_to=NOW-timedelta(days=10))
        future_end = ordinary("future-end", days=120, valid_to=NOW+timedelta(days=10))
        future_start = ordinary("future-start", days=120, valid_from=NOW+timedelta(days=1))
        snapshot = MemorySnapshot(0, [future_start, future_end, recent_end, old_end])
        plan = MemoryRetentionPolicy().plan(snapshot, "alice", NOW)
        self.assertEqual(plan.candidates, [MemorySourceRef(old_end.id, old_end.version)])

    def test_ordinary_archive_shares_protection_and_does_not_clean_other_layers_or_users(self):
        pinned = ordinary("pinned", metadata={"pinned": True})
        cited, released = ordinary("cited"), ordinary("released")
        summary = record("summary", layer="summary", source_refs=[MemorySourceRef(cited.id, cited.version)])
        stale = record("stale", layer="summary", summary_state="stale",
                       source_refs=[MemorySourceRef(released.id, released.version)])
        working = ordinary("working", layer="working", task_id="task-1")
        old_summary = ordinary("old-summary", layer="summary")
        foreign = ordinary("foreign", user_id="bob")
        foreign_summary = record("bob-summary", user_id="bob", layer="summary",
                                 source_refs=[MemorySourceRef(released.id, released.version)])
        both = ordinary("both", metadata={"pinned": True})
        summary.source_refs.append(MemorySourceRef(both.id, both.version))
        plan = MemoryRetentionPolicy().plan(MemorySnapshot(0, [pinned, cited, released, summary, stale,
                                                                  working, old_summary, foreign, foreign_summary, both]),
                                             "alice", NOW)
        self.assertEqual(plan.candidates, [MemorySourceRef(released.id, released.version)])
        self.assertEqual(plan.protected, {pinned.id: "pinned", cited.id: "summary_reference", both.id: "pinned"})

    def test_mixed_fact_and_ordinary_candidates_share_one_chronological_batch(self):
        fact = expired("b", days=110)
        experience = ordinary("a", days=110)
        later = ordinary("later", days=100)
        policy = MemoryRetentionPolicy(MemoryConfig(history_cleanup_batch_size=2))
        snapshot = MemorySnapshot(0, [later, fact, experience])
        before = deepcopy(snapshot)
        plan = policy.plan(snapshot, "alice", NOW)
        self.assertEqual(plan.candidates, [MemorySourceRef(experience.id, experience.version),
                                          MemorySourceRef(fact.id, fact.version)])
        self.assertEqual(policy.plan(snapshot, "alice", NOW, limit=0).candidates, [])
        self.assertEqual(snapshot, before)

    def test_invalid_plan_arguments_raise_clear_errors(self):
        policy = MemoryRetentionPolicy()
        base = {"snapshot": MemorySnapshot(), "user_id": "alice", "now": NOW}
        cases = [{"user_id": value} for value in (None, "", " ", 1)]
        cases += [{"limit": value} for value in (-1, True, 1.5, "2")]
        cases += [{"now": value} for value in (None, "now", NOW.replace(tzinfo=None),
                                                datetime(1, 1, 1, tzinfo=timezone.utc))]
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                policy.plan(**{**base, **changes})

    def test_corrupt_snapshots_are_rejected_before_a_cleanup_plan_is_returned(self):
        policy = MemoryRetentionPolicy()
        cases = [None, {}, MemorySnapshot(revision=True), MemorySnapshot(revision=-1),
                 MemorySnapshot(records=()), MemorySnapshot(records=[{}]),
                 MemorySnapshot(records=[ordinary(updated_at=NOW.replace(tzinfo=None))]),
                 MemorySnapshot(records=[ordinary(metadata={"bad": object()})]),
                 MemorySnapshot(records=[ordinary(version=0)]),
                 MemorySnapshot(records=[record(layer="summary", source_refs=[{}])]),
                 MemorySnapshot(records=[ordinary(), ordinary()])]
        for snapshot in cases:
            with self.subTest(snapshot=snapshot), self.assertRaises(ValueError):
                policy.plan(snapshot, "alice", NOW)

    def test_equivalent_timezones_produce_the_same_utc_plan_and_leave_input_unchanged(self):
        local_zone = timezone(timedelta(hours=8))
        experience = ordinary(updated_at=(NOW-timedelta(days=90)).astimezone(local_zone))
        snapshot = MemorySnapshot(9, [experience])
        before = deepcopy(snapshot)
        policy = MemoryRetentionPolicy()
        utc_plan = policy.plan(snapshot, "alice", NOW)
        local_plan = policy.plan(snapshot, "alice", NOW.astimezone(local_zone))
        self.assertEqual(local_plan, utc_plan)
        self.assertEqual(local_plan.checked_at.tzinfo, timezone.utc)
        self.assertEqual(local_plan.cutoff.tzinfo, timezone.utc)
        self.assertEqual(snapshot, before)
        local_plan.candidates[0].version = 99
        self.assertEqual(snapshot, before)

    def test_only_literal_true_is_pinned_and_read_only_planning_does_not_access_files(self):
        pinned = ordinary("pinned", metadata={"pinned": True})
        numeric = ordinary("numeric", metadata={"pinned": 1})
        textual = ordinary("textual", metadata={"pinned": "true"})
        snapshot = MemorySnapshot(0, [pinned, numeric, textual])
        before = deepcopy(snapshot)
        with patch("context_lab.storage.json_store.JsonFileStore.read", side_effect=AssertionError("计划不能读文件")), \
             patch("context_lab.storage.json_store.JsonFileStore.write", side_effect=AssertionError("计划不能写文件")):
            plan = MemoryRetentionPolicy().plan(snapshot, "alice", NOW)
        self.assertEqual(plan.protected, {pinned.id: "pinned"})
        self.assertEqual([ref.memory_id for ref in plan.candidates], [numeric.id, textual.id])
        self.assertEqual(snapshot, before)

    def test_preview_does_not_delete_or_append_events(self):
        service, old, current, pinned = self.seed()
        before = service.path.read_bytes()
        report = service.cleanup_history()
        self.assertTrue(report.dry_run)
        self.assertEqual([ref.memory_id for ref in report.plan.candidates], [old.id])
        self.assertEqual(report.deleted_ids, [])
        self.assertEqual(report.index_event_ids, [])
        self.assertIsNone(report.committed_revision)
        self.assertEqual(service.path.read_bytes(), before)
        self.assertIs(service.last_cleanup_report, report)

    def test_apply_preserves_other_users_current_facts_links_and_existing_outbox(self):
        service, old, current, pinned = self.seed()
        bob = self.service("bob")
        foreign = expired("foreign", user_id="bob")
        bob.repository.commit(MemorySnapshot(1, [foreign]))
        report = service.cleanup_history(dry_run=False)
        self.assertEqual(report.deleted_ids, [old.id])
        self.assertEqual(report.committed_revision, 3)
        self.assertEqual(service.repository.read_snapshot().records, [current, pinned])
        self.assertEqual(bob.repository.read_snapshot().records, [foreign])
        self.assertEqual(service.get(current.id).supersedes_id, old.id)
        events = service.repository.read_snapshot().index_events
        self.assertEqual(events[0].id, "old-upsert")
        deletion = [event for event in events if event.operation == "delete"]
        self.assertEqual([(event.memory_id, event.version) for event in deletion], [(old.id, old.version+1)])
        self.assertEqual(report.index_event_ids, [deletion[0].id])

    def test_no_candidates_do_not_create_a_business_file_or_commit(self):
        service = self.service()
        with patch.object(service.repository, "commit", side_effect=AssertionError("零候选不能提交")):
            report = service.cleanup_history(dry_run=False)
        self.assertEqual(report.deleted_ids, [])
        self.assertEqual(report.plan.candidates, [])
        self.assertIsNone(report.committed_revision)
        self.assertFalse(service.path.exists())

    def test_apply_removes_ordinary_archive_and_enqueues_versioned_delete_in_same_commit(self):
        service, old, current, pinned = self.seed()
        experience = ordinary("old-experience", days=100, version=3)
        snapshot = service.repository.read_snapshot()
        snapshot.records.append(experience)
        service.repository.commit(snapshot)
        report = service.cleanup_history(dry_run=False)
        self.assertEqual(report.deleted_ids, [experience.id, old.id])
        self.assertEqual(report.committed_revision, 3)
        saved = service.repository.read_snapshot()
        self.assertEqual(saved.records, [current, pinned])
        self.assertEqual(saved.index_events[0].id, "old-upsert")
        deletions = [event for event in saved.index_events if event.operation == "delete"]
        self.assertEqual([(event.memory_id, event.version) for event in deletions],
                         [(experience.id, 4), (old.id, old.version+1)])
        self.assertEqual(report.index_event_ids, [event.id for event in deletions])

    def test_conflict_replans_and_keeps_ordinary_archive_updated_by_another_instance(self):
        service = self.service()
        experience = ordinary("experience")
        service.repository.commit(MemorySnapshot(0, [experience]))
        other = self.service()

        def competing_update(snapshot):
            other.update(experience.id, content="刚补充了新的排查结论")
            raise MemoryConflict("test conflict")

        with patch.object(service.repository, "commit", side_effect=competing_update) as commit:
            report = service.cleanup_history(dry_run=False)
        self.assertEqual(commit.call_count, 1)
        self.assertEqual(report.deleted_ids, [])
        self.assertEqual(report.plan.candidates, [])
        self.assertIsNone(report.committed_revision)
        saved = service.get(experience.id)
        self.assertEqual(saved.updated_at, NOW)
        self.assertEqual(saved.version, experience.version+1)
        self.assertFalse(any(event.operation == "delete" for event in service.repository.read_snapshot().index_events))

    def test_conflict_replans_when_a_new_summary_starts_protecting_a_candidate(self):
        service, old, current, pinned = self.seed()
        other = self.service()

        def competing_reference(snapshot):
            latest = other.repository.read_snapshot()
            latest.records.append(record("new-summary", layer="summary", source_refs=[MemorySourceRef(old.id, old.version)]))
            other.repository.commit(latest)
            raise MemoryConflict("test conflict")

        with patch.object(service.repository, "commit", side_effect=competing_reference) as commit:
            report = service.cleanup_history(dry_run=False)
        self.assertEqual(commit.call_count, 1)
        self.assertEqual(report.deleted_ids, [])
        self.assertEqual(report.plan.protected[old.id], "summary_reference")
        self.assertEqual(service.get(old.id), old)
        self.assertIsNone(report.committed_revision)

    def test_write_failure_preserves_history_outbox_and_last_report(self):
        service, old, current, pinned = self.seed()
        before = service.path.read_bytes()
        with patch.object(service.repository.store, "write", side_effect=OSError("test write failed")) as write:
            with self.assertRaises(OSError):
                service.cleanup_history(dry_run=False)
        self.assertEqual(write.call_count, 1)
        self.assertEqual(service.path.read_bytes(), before)
        self.assertEqual(service.get(old.id), old)
        self.assertIsNone(service.last_cleanup_report)

    def test_invalid_cleanup_arguments_are_rejected_before_reading(self):
        service = self.service()
        cases = [{"dry_run": value} for value in (None, 0, "true")]
        cases += [{"limit": value} for value in (-1, True, 1.5, "2")]
        with patch.object(service.repository, "read_snapshot", side_effect=AssertionError("非法参数不能读取")):
            for changes in cases:
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    service.cleanup_history(**changes)

    def test_cleanup_limit_is_shared_by_both_types_and_zero_does_not_commit(self):
        service, old, current, pinned = self.seed()
        experience = ordinary("older-experience", days=100)
        snapshot = service.repository.read_snapshot()
        snapshot.records.append(experience)
        service.repository.commit(snapshot)
        before = service.path.read_bytes()
        with patch.object(service.repository, "commit", side_effect=AssertionError("零上限不能提交")):
            report = service.cleanup_history(dry_run=False, limit=0)
        self.assertEqual(report.deleted_ids, [])
        self.assertEqual(service.path.read_bytes(), before)
        report = service.cleanup_history(dry_run=False, limit=1)
        self.assertEqual(report.deleted_ids, [experience.id])
        self.assertEqual(service.get(old.id), old)

    def test_write_failure_preserves_the_previous_successful_report(self):
        service, old, current, pinned = self.seed()
        previous = service.cleanup_history()
        before = service.path.read_bytes()
        with patch.object(service.repository.store, "write", side_effect=OSError("test write failed")):
            with self.assertRaises(OSError):
                service.cleanup_history(dry_run=False)
        self.assertIs(service.last_cleanup_report, previous)
        self.assertEqual(previous.deleted_ids, [])
        self.assertEqual(service.path.read_bytes(), before)

    def test_read_failure_is_not_hidden_or_published_as_an_empty_success(self):
        service, old, current, pinned = self.seed()
        previous = service.cleanup_history()
        with patch.object(service.repository, "read_snapshot", side_effect=OSError("test read failed")):
            with self.assertRaises(OSError):
                service.cleanup_history(dry_run=False)
        self.assertIs(service.last_cleanup_report, previous)

    def test_conflicts_stop_after_three_attempts_without_publishing_a_success(self):
        service, old, current, pinned = self.seed()
        previous = service.cleanup_history()
        before = service.path.read_bytes()
        with patch.object(service.repository, "commit", side_effect=MemoryConflict("test conflict")) as commit:
            with self.assertRaises(MemoryConflict):
                service.cleanup_history(dry_run=False)
        self.assertEqual(commit.call_count, 3)
        self.assertEqual(service.path.read_bytes(), before)
        self.assertIs(service.last_cleanup_report, previous)
        self.assertEqual(previous.deleted_ids, [])

    def test_retry_reuses_event_identity_but_only_persists_one_deletion(self):
        service, old, current, pinned = self.seed()
        original_commit = service.repository.commit
        attempted_ids = []

        def once_conflict(snapshot):
            deletion = [event for event in snapshot.index_events if event.operation == "delete"]
            attempted_ids.append(deletion[0].id)
            if len(attempted_ids) == 1:
                raise MemoryConflict("test conflict")
            return original_commit(snapshot)

        with patch.object(service.repository, "commit", side_effect=once_conflict):
            report = service.cleanup_history(dry_run=False)
        self.assertEqual(attempted_ids, report.index_event_ids * 2)
        self.assertEqual(report.deleted_ids, [old.id])
        self.assertEqual(len([event for event in service.repository.read_snapshot().index_events
                              if event.operation == "delete"]), 1)

    def test_conflict_replans_when_another_instance_pins_a_fact(self):
        service, old, current, pinned = self.seed()
        other = self.service()

        def competing_pin(snapshot):
            other.update(old.id, metadata={**old.metadata, "pinned": True})
            raise MemoryConflict("test conflict")

        with patch.object(service.repository, "commit", side_effect=competing_pin) as commit:
            report = service.cleanup_history(dry_run=False)
        self.assertEqual(commit.call_count, 1)
        self.assertEqual(report.deleted_ids, [])
        self.assertEqual(report.plan.protected[old.id], "pinned")
        self.assertTrue(service.get(old.id).metadata["pinned"])

    def test_enabled_worker_runs_real_cleanup_once_and_respects_the_interval(self):
        service, old, current, pinned = self.seed()
        enabled = MemoryService(service.path, "alice", clock=lambda: NOW,
                                config=MemoryConfig(auto_cleanup_history=True))
        report = enabled.history_cleanup_worker.tick()
        self.assertEqual(report.deleted_ids, [old.id])
        self.assertEqual(enabled.repository.read_snapshot().records, [current, pinned])
        self.assertIsNone(enabled.history_cleanup_worker.tick())
        self.assertIs(enabled.last_cleanup_report, report)
