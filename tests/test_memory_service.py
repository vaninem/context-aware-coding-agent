"""已完成的 Service CRUD 检查；检索、事实修订和摘要生成继续留作练习。"""

from copy import deepcopy
from datetime import timedelta, timezone
from unittest.mock import Mock, patch

from context_lab.memory import MemoryConflict, MemoryIndexEvent, MemoryQuery, MemoryService, MemorySourceRef
from context_lab.models import ScriptedChatModel
from tests.memory_support import NOW, RecordingMemoryIndex, record
from tests.support import SandboxTestCase


class MemoryServiceAddTests(SandboxTestCase):
    def service(self, user_id="alice", **kwargs):
        return MemoryService(self.tmp / "memory.json", user_id,
                             clock=kwargs.pop("clock", lambda: NOW), **kwargs)

    def test_adds_persist_records_and_matching_events_without_resetting_old_times(self):
        service = self.service()
        first = service.add("first")
        service.clock = lambda: NOW + timedelta(hours=1)
        second = service.add("second", layer="working", task_id="task-1", importance=0)
        saved = self.service().repository.read_snapshot()
        self.assertEqual(saved.revision, 2)
        self.assertEqual(saved.records, [first, second])
        self.assertNotEqual(first.id, second.id)
        self.assertEqual(saved.records[0].created_at, NOW)
        self.assertEqual(saved.records[0].updated_at, NOW)
        self.assertEqual(second.created_at, NOW + timedelta(hours=1))
        self.assertEqual(second.updated_at, second.created_at)
        self.assertEqual(first.metadata, {})
        self.assertEqual(first.source_refs, [])
        self.assertIsNone(first.supersedes_id)
        self.assertEqual(first.summary_state, "fresh")
        self.assertEqual(second.importance, 0)
        self.assertEqual(len({event.id for event in saved.index_events}), 2)
        for item, event in zip(saved.records, saved.index_events):
            self.assertEqual((event.user_id, event.memory_id, event.operation, event.version),
                             ("alice", item.id, "upsert", item.version))
            self.assertEqual(event.created_at, item.created_at)
            self.assertEqual(event.attempts, 0)

    def test_input_return_and_persisted_metadata_are_independent(self):
        service = self.service()
        metadata = {"tags": ["original"]}
        item = service.add("remember", metadata=metadata)
        metadata["tags"].append("caller change")
        self.assertEqual(item.metadata, {"tags": ["original"]})
        item.metadata["tags"].append("return change")
        self.assertEqual(service.repository.read_snapshot().records[0].metadata, {"tags": ["original"]})

    def test_two_users_keep_their_records_and_events_across_restarts(self):
        a = self.service().add("alice first")
        b = self.service("bob").add("bob first")
        c = self.service().add("alice second")
        alice = self.service().repository.read_snapshot()
        bob = self.service("bob").repository.read_snapshot()
        self.assertEqual(alice.records, [a, c])
        self.assertEqual(bob.records, [b])
        self.assertEqual([event.memory_id for event in alice.index_events], [a.id, c.id])
        self.assertEqual([event.memory_id for event in bob.index_events], [b.id])
        self.assertEqual(alice.revision, bob.revision)

    def test_add_only_saves_outbox_without_calling_embedding_or_summary_model(self):
        index, model = RecordingMemoryIndex(), ScriptedChatModel([])
        service = self.service(embedding=index, summary_model=model)
        service.add("remember")
        self.assertEqual(index.applied, [])
        self.assertEqual(index.search_calls, [])
        self.assertEqual(index.process_calls, [])
        self.assertEqual(model.calls, [])
        self.assertEqual(len(service.repository.read_snapshot().index_events), 1)

    def test_conflict_reloads_new_data_and_reuses_ids_and_creation_time(self):
        clock = Mock(return_value=NOW)
        service, other = self.service(clock=clock), self.service()
        original_commit = service.repository.commit
        attempts = []

        def competing_commit(snapshot):
            attempts.append((snapshot.records[-1].id, snapshot.index_events[-1].id, snapshot.revision))
            if len(attempts) == 1:
                other.add("concurrent memory")
                raise MemoryConflict("test conflict")
            return original_commit(snapshot)

        with patch.object(service.repository, "commit", side_effect=competing_commit):
            item = service.add("requested memory")
        saved = service.repository.read_snapshot()
        self.assertEqual([record.content for record in saved.records], ["concurrent memory", "requested memory"])
        self.assertEqual(saved.records[-1], item)
        self.assertEqual(len(saved.index_events), 2)
        self.assertEqual(attempts[0][:2], attempts[1][:2])
        self.assertEqual([attempt[2] for attempt in attempts], [0, 1])
        clock.assert_called_once_with()

    def test_repeated_conflicts_have_a_finite_limit_and_do_not_save_duplicates(self):
        clock = Mock(return_value=NOW)
        service = self.service(clock=clock)
        with patch.object(service.repository, "commit", side_effect=MemoryConflict("test conflict")) as commit:
            with self.assertRaises(MemoryConflict):
                service.add("remember")
        self.assertEqual(commit.call_count, 3)
        self.assertEqual(len({call.args[0].records[-1].id for call in commit.call_args_list}), 1)
        self.assertEqual(len({call.args[0].index_events[-1].id for call in commit.call_args_list}), 1)
        clock.assert_called_once_with()
        self.assertFalse(service.path.exists())

    def test_write_errors_are_not_retried_and_preserve_existing_records_and_events(self):
        service = self.service()
        service.add("existing")
        before = service.path.read_bytes()
        metadata = {"tags": ["original"]}
        original = deepcopy(metadata)
        with patch.object(service.repository.store, "write", side_effect=OSError("test write failed")) as write:
            with self.assertRaises(OSError):
                service.add("cannot save", metadata=metadata)
        self.assertEqual(write.call_count, 1)
        self.assertEqual(service.path.read_bytes(), before)
        self.assertEqual(metadata, original)

    def test_fact_intervals_allow_adjacent_periods_but_reject_overlaps(self):
        service = self.service()
        end = NOW + timedelta(days=1)
        service.add("first", fact_key="project.database", valid_from=NOW, valid_to=end)
        before = service.path.read_bytes()
        for bounds in ({}, {"valid_from": NOW + timedelta(hours=1)}, {"valid_to": end}):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                service.add("overlaps", fact_key="project.database", **bounds)
            self.assertEqual(service.path.read_bytes(), before)
        service.add("adjacent future", fact_key="project.database", valid_from=end)
        service.add("adjacent history", fact_key="project.database", valid_to=NOW)
        service.add("different fact", fact_key="project.language", valid_from=NOW)
        self.service("bob").add("different user", fact_key="project.database", valid_from=NOW)
        self.assertEqual(len(service.repository.read_snapshot().records), 4)

    def test_fact_overlap_is_rechecked_after_a_version_conflict(self):
        service, other = self.service(), self.service()

        def competing_commit(snapshot):
            other.add("concurrent fact", fact_key="project.database", valid_from=NOW)
            raise MemoryConflict("test conflict")

        with patch.object(service.repository, "commit", side_effect=competing_commit) as commit:
            with self.assertRaises(ValueError):
                service.add("requested fact", fact_key="project.database", valid_from=NOW)
        self.assertEqual(commit.call_count, 1)
        saved = service.repository.read_snapshot()
        self.assertEqual([item.content for item in saved.records], ["concurrent fact"])
        self.assertEqual(len(saved.index_events), 1)

    def test_bad_inputs_do_not_create_a_business_file(self):
        cases = [
            {"content": " "}, {"content": 3}, {"memory_type": ""}, {"metadata": []},
            {"metadata": {"bad": float("nan")}}, {"importance": True},
            {"importance": -1}, {"importance": float("inf")},
            {"layer": "other"}, {"layer": "summary"}, {"layer": "working"},
            {"fact_key": " "}, {"valid_from": NOW.replace(tzinfo=None)},
            {"valid_from": NOW, "valid_to": NOW},
        ]
        service = self.service()
        for arguments in cases:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                service.add(**{"content": "remember", **arguments})
            self.assertFalse(service.path.exists())
        with self.assertRaises(ValueError):
            self.service(clock=lambda: NOW.replace(tzinfo=None)).add("remember")


class MemoryServiceGetTests(SandboxTestCase):
    def service(self, user_id="alice"):
        return MemoryService(self.tmp / "memory.json", user_id, clock=lambda: NOW)

    def test_get_recovers_after_restart_and_returns_an_independent_read_only_copy(self):
        service = self.service()
        original = service.add("remember", metadata={"tags": ["original"]})
        before = service.path.read_bytes()
        reader = self.service()
        with patch.object(reader.repository, "commit", side_effect=AssertionError("get 不能提交")):
            with patch.object(reader, "clock", side_effect=AssertionError("get 不能修改时间")):
                returned = reader.get(original.id)
        self.assertEqual(returned, original)
        returned.content = "local change"
        returned.metadata["tags"].append("local tag")
        self.assertEqual(original.content, "remember")
        self.assertEqual(original.metadata, {"tags": ["original"]})
        self.assertEqual(reader.get(original.id), original)
        self.assertEqual(service.path.read_bytes(), before)

    def test_missing_foreign_and_invalid_ids_do_not_expose_other_users_or_create_files(self):
        service = self.service()
        for memory_id in (None, "", " ", 123, [], "missing"):
            with self.subTest(memory_id=memory_id), self.assertRaises(ValueError):
                service.get(memory_id)
            self.assertFalse(service.path.exists())
        bob_item = self.service("bob").add("bob only")
        before = service.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "找不到记忆"):
            service.get(bob_item.id)
        self.assertEqual(self.service("bob").get(bob_item.id), bob_item)
        self.assertEqual(service.path.read_bytes(), before)

    def test_get_reads_expired_evidence_and_stale_summaries_by_id(self):
        service = self.service()
        original = service.add("historical evidence", valid_to=NOW)
        snapshot = service.repository.read_snapshot()
        summary = record("summary-1", layer="summary", summary_state="stale", valid_to=NOW,
                         source_refs=[MemorySourceRef(original.id, original.version)])
        snapshot.records.append(summary)
        snapshot.index_events.append(MemoryIndexEvent("summary-delete", "alice", summary.id, "delete", 1, NOW))
        service.repository.commit(snapshot)
        before = service.path.read_bytes()
        restored = service.get(summary.id)
        self.assertEqual(service.get(original.id), original)
        self.assertEqual(restored, summary)
        restored.source_refs[0].version = 9
        self.assertEqual(service.get(summary.id).source_refs[0].version, 1)
        self.assertEqual(service.path.read_bytes(), before)

    def test_file_errors_are_not_misreported_as_missing_memory(self):
        service = self.service()
        with patch.object(service.repository.store, "read", side_effect=PermissionError("test read denied")):
            with self.assertRaises(PermissionError):
                service.get("memory-1")
        service.path.write_text('{"broken":', encoding="utf-8")
        with self.assertRaises(ValueError) as caught:
            service.get("memory-1")
        self.assertNotIn("找不到记忆", str(caught.exception))


class MemoryServiceUpdateTests(SandboxTestCase):
    def service(self, user_id="alice"):
        return MemoryService(self.tmp / "memory.json", user_id, clock=lambda: NOW)

    def test_updates_fields_preserves_identity_and_returns_independent_persisted_values(self):
        service = self.service()
        original = service.add("before", importance=0.8, metadata={"old": True})
        service.clock = lambda: NOW + timedelta(hours=1)
        metadata = {"tags": ["new"]}
        updated = service.update(original.id, content="after", importance=0, metadata=metadata)
        self.assertEqual((updated.id, updated.user_id, updated.created_at),
                         (original.id, original.user_id, original.created_at))
        self.assertEqual(updated.updated_at, NOW + timedelta(hours=1))
        self.assertEqual(updated.version, original.version + 1)
        self.assertEqual((updated.content, updated.importance, updated.metadata),
                         ("after", 0, {"tags": ["new"]}))
        saved = self.service().repository.read_snapshot()
        self.assertEqual(saved.records, [updated])
        self.assertEqual((saved.index_events[-1].memory_id, saved.index_events[-1].version,
                          saved.index_events[-1].operation), (updated.id, updated.version, "upsert"))
        metadata["tags"].append("caller change")
        self.assertEqual(updated.metadata, {"tags": ["new"]})
        updated.metadata["tags"].append("return change")
        self.assertEqual(service.get(original.id).metadata, {"tags": ["new"]})
        cleared = service.update(original.id, metadata={})
        self.assertEqual(cleared.metadata, {})
        self.assertEqual((cleared.content, cleared.importance), ("after", 0))

    def test_invalidates_only_related_fresh_summaries_and_uses_matching_event_versions(self):
        service = self.service()
        raw, other = service.add("source"), service.add("unrelated source")
        snapshot = service.repository.read_snapshot()
        related = record("related", layer="summary", source_refs=[MemorySourceRef(raw.id, raw.version)])
        unrelated = record("unrelated", layer="summary", source_refs=[MemorySourceRef(other.id, other.version)])
        stale = record("stale", layer="summary", summary_state="stale", version=5,
                       source_refs=[MemorySourceRef(raw.id, raw.version)])
        snapshot.records.extend([related, unrelated, stale])
        service.repository.commit(snapshot)
        bob = self.service("bob")
        bob_snapshot = bob.repository.read_snapshot()
        bob_source = record(raw.id, user_id="bob")
        bob_summary = record("bob-summary", user_id="bob", layer="summary",
                             source_refs=[MemorySourceRef(raw.id, 1)])
        bob_snapshot.records.extend([bob_source, bob_summary])
        bob.repository.commit(bob_snapshot)
        service.clock = lambda: NOW + timedelta(hours=1)
        updated = service.update(raw.id, content="changed source")
        saved_summary = service.get(related.id)
        self.assertEqual((saved_summary.summary_state, saved_summary.version), ("stale", 2))
        self.assertEqual(saved_summary.updated_at, NOW + timedelta(hours=1))
        self.assertEqual(saved_summary.source_refs, related.source_refs)
        self.assertEqual(service.get(unrelated.id), unrelated)
        self.assertEqual(service.get(stale.id), stale)
        self.assertEqual(bob.get(bob_summary.id), bob_summary)
        saved = service.repository.read_snapshot()
        deletion = [event for event in saved.index_events if event.operation == "delete"]
        self.assertEqual([(event.memory_id, event.version) for event in deletion], [(related.id, 2)])
        target_event = next(event for event in saved.index_events
                            if event.memory_id == raw.id and event.version == updated.version)
        self.assertEqual(target_event.operation, "upsert")
        service.update(raw.id, content="changed again")
        self.assertEqual(service.get(related.id), saved_summary)
        self.assertEqual(len([event for event in service.repository.read_snapshot().index_events
                              if event.operation == "delete"]), 1)

    def test_fact_body_changes_are_rejected_but_metadata_and_importance_can_change(self):
        service = self.service()
        item = service.add("database is PostgreSQL", fact_key="project.database", valid_from=NOW)
        before = service.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "revise_fact"):
            service.update(item.id, content="database is SQLite")
        self.assertEqual(service.path.read_bytes(), before)
        changed = service.update(item.id, content=item.content, importance=0, metadata={"source": "config"})
        self.assertEqual((changed.content, changed.fact_key, changed.valid_from),
                         (item.content, item.fact_key, item.valid_from))
        self.assertEqual(changed.importance, 0)

    def test_empty_or_equal_updates_do_not_consume_versions_or_invalidate_summaries(self):
        service = self.service()
        item = service.add("remember", importance=0)
        snapshot = service.repository.read_snapshot()
        summary = record("summary", layer="summary", source_refs=[MemorySourceRef(item.id, 1)])
        snapshot.records.append(summary)
        service.repository.commit(snapshot)
        before = service.path.read_bytes()
        with patch.object(service.repository, "commit", side_effect=AssertionError("无变化不能提交")):
            self.assertEqual(service.update(item.id), item)
            self.assertEqual(service.update(item.id, content=item.content, importance=0, metadata={}), item)
        self.assertEqual(service.get(summary.id).summary_state, "fresh")
        self.assertEqual(service.path.read_bytes(), before)

    def test_conflict_reloads_latest_fields_and_reuses_the_event_id(self):
        service, other = self.service(), self.service()
        item = service.add("before", importance=0.5)
        original_commit = service.repository.commit
        attempts = []

        def competing_commit(snapshot):
            target = next(record for record in snapshot.records if record.id == item.id)
            attempts.append((target.version, snapshot.index_events[-1].id))
            if len(attempts) == 1:
                other.update(item.id, importance=0.9)
                raise MemoryConflict("test conflict")
            return original_commit(snapshot)

        with patch.object(service.repository, "commit", side_effect=competing_commit):
            updated = service.update(item.id, content="after")
        self.assertEqual((updated.content, updated.importance, updated.version), ("after", 0.9, 3))
        self.assertEqual([version for version, event_id in attempts], [2, 3])
        self.assertEqual(attempts[0][1], attempts[1][1])
        self.assertEqual(len(service.repository.read_snapshot().index_events), 3)

    def test_repeated_conflicts_stop_without_changing_file(self):
        service = self.service()
        item = service.add("before")
        before = service.path.read_bytes()
        with patch.object(service.repository, "commit", side_effect=MemoryConflict("test conflict")) as commit:
            with self.assertRaises(MemoryConflict):
                service.update(item.id, content="after")
        self.assertEqual(commit.call_count, 3)
        self.assertEqual(len({call.args[0].index_events[-1].id for call in commit.call_args_list}), 1)
        self.assertEqual(service.path.read_bytes(), before)

    def test_write_failure_preserves_record_summary_and_outbox_as_one_unit(self):
        service = self.service()
        item = service.add("before")
        snapshot = service.repository.read_snapshot()
        snapshot.records.append(record("summary", layer="summary", source_refs=[MemorySourceRef(item.id, 1)]))
        service.repository.commit(snapshot)
        before = service.path.read_bytes()
        metadata = {"tags": ["new"]}
        with patch.object(service.repository.store, "write", side_effect=OSError("test write failed")) as write:
            with self.assertRaises(OSError):
                service.update(item.id, content="after", metadata=metadata)
        self.assertEqual(write.call_count, 1)
        self.assertEqual(service.path.read_bytes(), before)
        self.assertEqual(metadata, {"tags": ["new"]})
        self.assertEqual(service.get("summary").summary_state, "fresh")

    def test_invalid_missing_and_foreign_updates_cannot_change_data(self):
        service, bob = self.service(), self.service("bob")
        item = service.add("remember", importance=0)
        bob_item = bob.add("bob only")
        before = service.path.read_bytes()
        for memory_id in (None, "", " ", [], "missing", bob_item.id):
            with self.subTest(memory_id=memory_id), self.assertRaises(ValueError):
                service.update(memory_id, content="changed")
            self.assertEqual(service.path.read_bytes(), before)
        for changes in ({"content": " "}, {"importance": False}, {"importance": float("nan")},
                        {"metadata": []}, {"metadata": {"bad": float("inf")}}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                service.update(item.id, **changes)
            self.assertEqual(service.path.read_bytes(), before)

    def test_retry_does_not_resurrect_a_record_deleted_by_another_instance(self):
        service, other = self.service(), self.service()
        item = service.add("before")

        def delete_before_commit(snapshot):
            latest = other.repository.read_snapshot()
            latest.records = [record for record in latest.records if record.id != item.id]
            latest.index_events.append(MemoryIndexEvent("external-delete", "alice", item.id,
                                                       "delete", item.version + 1, NOW))
            other.repository.commit(latest)
            raise MemoryConflict("test conflict")

        with patch.object(service.repository, "commit", side_effect=delete_before_commit) as commit:
            with self.assertRaisesRegex(ValueError, "找不到记忆"):
                service.update(item.id, content="after")
        self.assertEqual(commit.call_count, 1)
        self.assertEqual(service.repository.read_snapshot().records, [])


class MemoryServiceRemoveTests(SandboxTestCase):
    def service(self, user_id="alice", **kwargs):
        return MemoryService(self.tmp / "memory.json", user_id,
                             clock=kwargs.pop("clock", lambda: NOW), **kwargs)

    def test_deletes_business_record_and_saves_outbox_without_calling_index(self):
        index = RecordingMemoryIndex()
        service = self.service(embedding=index)
        item = service.add("source")
        self.assertTrue(service.remove(item.id))
        saved = self.service().repository.read_snapshot()
        self.assertEqual(saved.records, [])
        self.assertEqual(saved.revision, 2)
        self.assertEqual([(event.memory_id, event.operation, event.version) for event in saved.index_events],
                         [(item.id, "upsert", 1), (item.id, "delete", 2)])
        self.assertEqual(index.applied, [])
        self.assertEqual(index.process_calls, [])
        before = service.path.read_bytes()
        self.assertFalse(service.remove(item.id))
        self.assertEqual(service.path.read_bytes(), before)

    def test_missing_foreign_and_invalid_ids_cannot_change_files(self):
        service = self.service()
        with patch.object(service, "clock", side_effect=AssertionError("不存在的删除不取时间")):
            self.assertFalse(service.remove("missing"))
        self.assertFalse(service.path.exists())
        for memory_id in (None, "", " ", 123, []):
            with self.subTest(memory_id=memory_id), self.assertRaises(ValueError):
                service.remove(memory_id)
            self.assertFalse(service.path.exists())
        alice_item = service.add("alice")
        bob = self.service("bob")
        bob_item = bob.add("bob")
        before = service.path.read_bytes()
        self.assertFalse(bob.remove(alice_item.id))
        self.assertEqual(service.path.read_bytes(), before)
        self.assertTrue(service.remove(alice_item.id))
        self.assertEqual(bob.get(bob_item.id), bob_item)

    def test_delete_invalidates_related_fresh_summaries_and_preserves_evidence_references(self):
        service = self.service()
        raw, other = service.add("source"), service.add("unrelated source")
        snapshot = service.repository.read_snapshot()
        related = record("related", layer="summary", source_refs=[MemorySourceRef(raw.id, 1)])
        unrelated = record("unrelated", layer="summary", source_refs=[MemorySourceRef(other.id, 1)])
        stale = record("stale", layer="summary", summary_state="stale", version=4,
                       source_refs=[MemorySourceRef(raw.id, 1)])
        snapshot.records.extend([related, unrelated, stale])
        service.repository.commit(snapshot)
        bob = self.service("bob")
        bob_snapshot = bob.repository.read_snapshot()
        bob_summary = record("bob-summary", user_id="bob", layer="summary",
                             source_refs=[MemorySourceRef(raw.id, 1)])
        bob_snapshot.records.extend([record(raw.id, user_id="bob"), bob_summary])
        bob.repository.commit(bob_snapshot)
        service.clock = lambda: NOW + timedelta(hours=1)
        self.assertTrue(service.remove(raw.id))
        summary = service.get(related.id)
        self.assertEqual((summary.summary_state, summary.version), ("stale", 2))
        self.assertEqual(summary.updated_at, NOW + timedelta(hours=1))
        self.assertEqual(summary.source_refs, related.source_refs)
        self.assertEqual(service.get(unrelated.id), unrelated)
        self.assertEqual(service.get(stale.id), stale)
        self.assertEqual(bob.get(bob_summary.id), bob_summary)
        deletions = [event for event in service.repository.read_snapshot().index_events
                     if event.operation == "delete"]
        self.assertEqual({(event.memory_id, event.version) for event in deletions},
                         {(raw.id, 2), (related.id, 2)})
        with self.assertRaises(ValueError):
            service.get(raw.id)

    def test_write_failure_keeps_source_summaries_and_events_unchanged(self):
        service = self.service()
        item = service.add("source")
        snapshot = service.repository.read_snapshot()
        snapshot.records.append(record("summary", layer="summary", source_refs=[MemorySourceRef(item.id, 1)]))
        service.repository.commit(snapshot)
        before = service.path.read_bytes()
        with patch.object(service.repository.store, "write", side_effect=OSError("test write failed")) as write:
            with self.assertRaises(OSError):
                service.remove(item.id)
        self.assertEqual(write.call_count, 1)
        self.assertEqual(service.path.read_bytes(), before)
        self.assertEqual(service.get(item.id), item)
        self.assertEqual(service.get("summary").summary_state, "fresh")

    def test_retry_deletes_latest_version_and_preserves_concurrent_new_memory(self):
        clock = Mock(return_value=NOW)
        service, other = self.service(clock=clock), self.service()
        item = service.add("source")
        clock.reset_mock()
        original_commit = service.repository.commit
        attempts = []

        def competing_commit(snapshot):
            event = next(event for event in snapshot.index_events
                         if event.memory_id == item.id and event.operation == "delete")
            attempts.append((event.id, event.version))
            if len(attempts) == 1:
                other.update(item.id, content="changed concurrently")
                other.add("concurrent new memory")
                raise MemoryConflict("test conflict")
            return original_commit(snapshot)

        with patch.object(service.repository, "commit", side_effect=competing_commit):
            self.assertTrue(service.remove(item.id))
        saved = service.repository.read_snapshot()
        self.assertEqual([record.content for record in saved.records], ["concurrent new memory"])
        self.assertEqual(attempts[0][0], attempts[1][0])
        self.assertEqual([version for event_id, version in attempts], [2, 3])
        clock.assert_called_once_with()

    def test_repeated_conflicts_stop_without_saving_delete_or_stale_summary(self):
        service = self.service()
        item = service.add("source")
        snapshot = service.repository.read_snapshot()
        snapshot.records.append(record("summary", layer="summary", source_refs=[MemorySourceRef(item.id, 1)]))
        service.repository.commit(snapshot)
        before = service.path.read_bytes()
        with patch.object(service.repository, "commit", side_effect=MemoryConflict("test conflict")) as commit:
            with self.assertRaises(MemoryConflict):
                service.remove(item.id)
        self.assertEqual(commit.call_count, 3)
        self.assertEqual(len({call.args[0].index_events[-1].id for call in commit.call_args_list}), 1)
        self.assertEqual(service.path.read_bytes(), before)

    def test_retry_returns_false_when_another_instance_already_deleted_the_record(self):
        service, other = self.service(), self.service()
        item = service.add("source")

        def competing_delete(snapshot):
            self.assertTrue(other.remove(item.id))
            raise MemoryConflict("test conflict")

        with patch.object(service.repository, "commit", side_effect=competing_delete) as commit:
            self.assertFalse(service.remove(item.id))
        self.assertEqual(commit.call_count, 1)
        saved = service.repository.read_snapshot()
        self.assertEqual(saved.records, [])
        self.assertEqual(len(saved.index_events), 2)

    def test_retry_also_invalidates_a_summary_created_after_the_first_read(self):
        service, other = self.service(), self.service()
        item = service.add("source")
        original_commit = service.repository.commit
        calls = []

        def competing_summary(snapshot):
            calls.append(snapshot.revision)
            if len(calls) == 1:
                latest = other.repository.read_snapshot()
                latest.records.append(record("new-summary", layer="summary",
                                             source_refs=[MemorySourceRef(item.id, item.version)]))
                latest.index_events.append(MemoryIndexEvent("summary-upsert", "alice", "new-summary",
                                                           "upsert", 1, NOW))
                other.repository.commit(latest)
                raise MemoryConflict("test conflict")
            return original_commit(snapshot)

        with patch.object(service.repository, "commit", side_effect=competing_summary):
            self.assertTrue(service.remove(item.id))
        summary = service.get("new-summary")
        self.assertEqual((summary.summary_state, summary.version), ("stale", 2))
        saved = service.repository.read_snapshot()
        self.assertEqual([record.id for record in saved.records], ["new-summary"])
        self.assertTrue(any(event.memory_id == summary.id and event.operation == "delete" and event.version == 2
                            for event in saved.index_events))


class MemoryServiceListTests(SandboxTestCase):
    def service(self, user_id="alice"):
        return MemoryService(self.tmp / "memory.json", user_id, clock=lambda: NOW)

    def test_default_listing_uses_service_clock_and_preserves_file_and_outbox(self):
        service = self.service()
        current = service.add("current", valid_from=NOW)
        service.add("future", valid_from=NOW + timedelta(hours=1))
        service.add("expired", valid_to=NOW)
        snapshot = service.repository.read_snapshot()
        snapshot.records.append(record("stale", layer="summary", summary_state="stale",
                                       source_refs=[MemorySourceRef(current.id, 1)]))
        service.repository.commit(snapshot)
        self.service("bob").add("bob only")
        before = service.path.read_bytes()
        service.clock = Mock(return_value=NOW)
        self.assertEqual(service.list_records(), [current])
        service.clock.assert_called_once_with()
        self.assertEqual(service.path.read_bytes(), before)

    def test_explicit_historical_filters_are_preserved_and_results_are_independent(self):
        service = self.service()
        old = service.add("SQLite", memory_type="semantic", importance=0, metadata={"tags": ["old"]},
                          fact_key="project.database", valid_from=NOW - timedelta(days=2), valid_to=NOW)
        service.add("PostgreSQL", memory_type="semantic", fact_key="project.database", valid_from=NOW)
        filters = MemoryQuery(as_of=NOW - timedelta(days=1), layers=("archive",), memory_type="semantic",
                              min_importance=0, max_importance=0)
        before = deepcopy(filters)
        listed = service.list_records(filters)
        self.assertEqual(listed, [old])
        self.assertEqual(filters, before)
        listed[0].metadata["tags"].append("local change")
        self.assertEqual(service.list_records(filters), [old])
        self.assertEqual(service.get(old.id).metadata, {"tags": ["old"]})
        self.assertEqual(service.list_records(MemoryQuery(layers=())), [])

    def test_missing_file_is_empty_but_bad_filters_and_file_errors_are_not_hidden(self):
        service = self.service()
        service.path = self.tmp / "absent" / "memory.json"
        service = MemoryService(service.path, "alice", clock=lambda: NOW)
        self.assertEqual(service.list_records(), [])
        self.assertFalse(service.path.parent.exists())
        for filters in ({}, [], False, "archive"):
            with self.subTest(filters=filters), self.assertRaises(ValueError):
                service.list_records(filters)
            self.assertFalse(service.path.parent.exists())
        with patch.object(service.repository.store, "read", side_effect=PermissionError("test read denied")):
            with self.assertRaises(PermissionError):
                service.list_records()
        service.path.parent.mkdir()
        service.path.write_text('{"broken":', encoding="utf-8")
        with self.assertRaises(ValueError):
            service.list_records()


class MemoryServiceReviseFactTests(SandboxTestCase):
    def service(self, user_id="alice"):
        return MemoryService(self.tmp / "memory.json", user_id, clock=lambda: NOW)

    def test_revision_keeps_old_body_and_saves_both_versions_and_matching_events(self):
        service = self.service()
        old = service.add("SQLite", memory_type="semantic", fact_key="project.database",
                          valid_from=NOW - timedelta(days=1), metadata={"tags": ["origin"]})
        boundary = NOW + timedelta(hours=1)
        operation_time = NOW + timedelta(days=1)
        service.clock = lambda: operation_time
        new = service.revise_fact(old.id, "PostgreSQL", boundary)
        saved_old = service.get(old.id)
        self.assertEqual((saved_old.content, saved_old.valid_from, saved_old.valid_to),
                         (old.content, old.valid_from, boundary))
        self.assertEqual((saved_old.created_at, saved_old.updated_at, saved_old.version),
                         (old.created_at, operation_time, 2))
        self.assertNotEqual(new.id, old.id)
        self.assertEqual((new.content, new.fact_key, new.supersedes_id, new.version),
                         ("PostgreSQL", old.fact_key, old.id, 1))
        self.assertEqual((new.valid_from, new.valid_to, new.created_at, new.updated_at),
                         (boundary, None, operation_time, operation_time))
        self.assertEqual(service.list_records(MemoryQuery(as_of=NOW)), [saved_old])
        self.assertEqual(service.list_records(MemoryQuery(as_of=boundary)), [new])
        saved = service.repository.read_snapshot()
        self.assertEqual(len(saved.records), 2)
        self.assertEqual([(event.memory_id, event.version, event.operation) for event in saved.index_events[-2:]],
                         [(old.id, 2, "upsert"), (new.id, 1, "upsert")])
        self.assertEqual(old.version, 1)
        self.assertIsNone(old.valid_to)
        new.metadata["tags"].append("local change")
        self.assertEqual(service.get(old.id).metadata, {"tags": ["origin"]})
        self.assertEqual(service.get(new.id).metadata, {"tags": ["origin"]})

    def test_bounded_revisions_preserve_future_facts_and_link_each_predecessor(self):
        service = self.service()
        start, middle, end = NOW - timedelta(days=3), NOW - timedelta(days=1), NOW + timedelta(days=2)
        first = service.add("SQLite", fact_key="project.database", valid_from=start, valid_to=end)
        planned = service.add("MySQL", fact_key="project.database", valid_from=end)
        second = service.revise_fact(first.id, "PostgreSQL", middle)
        third = service.revise_fact(second.id, "new database", NOW)
        self.assertEqual(second.valid_to, end)
        self.assertEqual(third.valid_to, end)
        self.assertEqual(third.supersedes_id, second.id)
        self.assertEqual(second.supersedes_id, first.id)
        self.assertEqual(service.get(planned.id), planned)
        self.assertEqual(service.get(second.id).valid_to, NOW)
        self.assertEqual(service.get(second.id).version, 2)
        self.assertEqual(third.version, 1)
        for when, expected in ((start, first.id), (middle, second.id), (NOW, third.id), (end, planned.id)):
            with self.subTest(when=when):
                self.assertEqual([item.id for item in service.list_records(MemoryQuery(as_of=when))], [expected])
        self.assertEqual(len(service.repository.read_snapshot().records), 4)

    def test_revision_invalidates_related_fresh_summaries_without_deleting_evidence(self):
        service = self.service()
        old = service.add("SQLite", fact_key="project.database", valid_from=NOW - timedelta(days=1))
        other = service.add("unrelated source")
        snapshot = service.repository.read_snapshot()
        related = record("related", layer="summary", source_refs=[MemorySourceRef(old.id, 1)])
        unrelated = record("unrelated", layer="summary", source_refs=[MemorySourceRef(other.id, 1)])
        stale = record("stale", layer="summary", summary_state="stale", version=5,
                       source_refs=[MemorySourceRef(old.id, 1)])
        snapshot.records.extend([related, unrelated, stale])
        service.repository.commit(snapshot)
        service.clock = lambda: NOW + timedelta(hours=1)
        service.revise_fact(old.id, "PostgreSQL", NOW)
        summary = service.get(related.id)
        self.assertEqual((summary.summary_state, summary.version, summary.updated_at),
                         ("stale", 2, NOW + timedelta(hours=1)))
        self.assertEqual(summary.source_refs, related.source_refs)
        self.assertEqual(service.get(unrelated.id), unrelated)
        self.assertEqual(service.get(stale.id), stale)
        self.assertEqual(service.get(old.id).content, "SQLite")
        deletions = [event for event in service.repository.read_snapshot().index_events if event.operation == "delete"]
        self.assertEqual([(event.memory_id, event.version) for event in deletions], [(related.id, 2)])

    def test_bad_ids_nonfacts_and_invalid_effective_times_cannot_change_file(self):
        service = self.service()
        start, end = NOW - timedelta(days=1), NOW + timedelta(days=1)
        old = service.add("SQLite", fact_key="project.database", valid_from=start, valid_to=end)
        ordinary = service.add("ordinary memory")
        foreign = self.service("bob").add("Bob fact", fact_key="project.database", valid_from=start)
        before = service.path.read_bytes()
        for memory_id in (None, " ", [], "missing", ordinary.id, foreign.id):
            with self.subTest(memory_id=memory_id), self.assertRaises(ValueError):
                service.revise_fact(memory_id, "new value", NOW)
            self.assertEqual(service.path.read_bytes(), before)
        for when in (None, "today", NOW.replace(tzinfo=None), start, end,
                     start - timedelta(seconds=1), end + timedelta(seconds=1)):
            with self.subTest(effective_from=when), self.assertRaises(ValueError):
                service.revise_fact(old.id, "new value", when)
            self.assertEqual(service.path.read_bytes(), before)
        for content in (None, " ", 123):
            with self.subTest(content=content), self.assertRaises(ValueError):
                service.revise_fact(old.id, content, NOW)
            self.assertEqual(service.path.read_bytes(), before)

    def test_unbounded_period_supports_offsets_and_keeps_recording_time_separate(self):
        service = self.service()
        old = service.add("SQLite", fact_key="project.database")
        service.clock = lambda: NOW + timedelta(days=1)
        local_boundary = NOW.astimezone(timezone(timedelta(hours=8)))
        new = service.revise_fact(old.id, "PostgreSQL", local_boundary)
        self.assertEqual(new.valid_from, NOW)
        self.assertEqual(new.valid_from.utcoffset(), timedelta(0))
        self.assertEqual(new.created_at, NOW + timedelta(days=1))
        self.assertIsNone(new.valid_to)
        self.assertIsNone(service.get(old.id).valid_from)
        self.assertEqual(service.get(old.id).created_at, NOW)
        self.assertEqual([item.id for item in service.list_records(MemoryQuery(as_of=NOW-timedelta(seconds=1)))], [old.id])
        self.assertEqual([item.id for item in service.list_records(MemoryQuery(as_of=NOW))], [new.id])

    def test_write_failure_preserves_original_fact_summary_and_all_events(self):
        service = self.service()
        old = service.add("SQLite", fact_key="project.database", valid_from=NOW - timedelta(days=1))
        snapshot = service.repository.read_snapshot()
        snapshot.records.append(record("summary", layer="summary", source_refs=[MemorySourceRef(old.id, 1)]))
        service.repository.commit(snapshot)
        before = service.path.read_bytes()
        with patch.object(service.repository.store, "write", side_effect=OSError("test write failed")) as write:
            with self.assertRaises(OSError):
                service.revise_fact(old.id, "PostgreSQL", NOW)
        self.assertEqual(write.call_count, 1)
        self.assertEqual(service.path.read_bytes(), before)
        self.assertEqual(service.get(old.id), old)
        self.assertEqual(service.get("summary").summary_state, "fresh")

    def test_conflict_inherits_latest_fields_and_reuses_new_id_and_event_ids(self):
        service, other = self.service(), self.service()
        old = service.add("SQLite", fact_key="project.database", valid_from=NOW - timedelta(days=1))
        clock = Mock(return_value=NOW)
        service.clock = clock
        original_commit = service.repository.commit
        attempts = []

        def competing_commit(snapshot):
            source = next(item for item in snapshot.records if item.id == old.id)
            new = next(item for item in snapshot.records if item.supersedes_id == old.id)
            source_event = next(event for event in snapshot.index_events
                                if event.memory_id == old.id and event.version == source.version)
            new_event = next(event for event in snapshot.index_events if event.memory_id == new.id)
            attempts.append((new.id, source_event.id, new_event.id, source.version))
            if len(attempts) == 1:
                other.update(old.id, importance=0.9, metadata={"source": "concurrent change"})
                other.add("concurrent new memory")
                raise MemoryConflict("test conflict")
            return original_commit(snapshot)

        with patch.object(service.repository, "commit", side_effect=competing_commit):
            new = service.revise_fact(old.id, "PostgreSQL", NOW)
        self.assertEqual(new.importance, 0.9)
        self.assertEqual(new.metadata, {"source": "concurrent change"})
        self.assertEqual(service.get(old.id).version, 3)
        self.assertEqual(attempts[0][:3], attempts[1][:3])
        self.assertEqual([attempt[3] for attempt in attempts], [2, 3])
        self.assertEqual(len(service.repository.read_snapshot().records), 3)
        clock.assert_called_once_with()

    def test_repeated_conflicts_stop_without_partially_closing_old_fact(self):
        service = self.service()
        old = service.add("SQLite", fact_key="project.database", valid_from=NOW - timedelta(days=1))
        before = service.path.read_bytes()
        with patch.object(service.repository, "commit", side_effect=MemoryConflict("test conflict")) as commit:
            with self.assertRaises(MemoryConflict):
                service.revise_fact(old.id, "PostgreSQL", NOW)
        self.assertEqual(commit.call_count, 3)
        new_ids = {next(item.id for item in call.args[0].records if item.supersedes_id == old.id)
                   for call in commit.call_args_list}
        self.assertEqual(len(new_ids), 1)
        self.assertEqual(service.path.read_bytes(), before)
        self.assertEqual(service.get(old.id), old)

    def test_retry_does_not_recreate_an_explicitly_deleted_source(self):
        service, other = self.service(), self.service()
        old = service.add("SQLite", fact_key="project.database", valid_from=NOW - timedelta(days=1))

        def delete_source(snapshot):
            self.assertTrue(other.remove(old.id))
            raise MemoryConflict("test conflict")

        with patch.object(service.repository, "commit", side_effect=delete_source) as commit:
            with self.assertRaisesRegex(ValueError, "找不到记忆"):
                service.revise_fact(old.id, "PostgreSQL", NOW)
        self.assertEqual(commit.call_count, 1)
        self.assertEqual(service.repository.read_snapshot().records, [])

    def test_retry_rechecks_interval_if_another_instance_already_revised_the_source(self):
        service, other = self.service(), self.service()
        old = service.add("SQLite", fact_key="project.database", valid_from=NOW - timedelta(days=1))

        def revise_source(snapshot):
            other.revise_fact(old.id, "concurrent database", NOW)
            raise MemoryConflict("test conflict")

        with patch.object(service.repository, "commit", side_effect=revise_source) as commit:
            with self.assertRaisesRegex(ValueError, "有效区间"):
                service.revise_fact(old.id, "requested database", NOW + timedelta(hours=1))
        self.assertEqual(commit.call_count, 1)
        self.assertEqual([item.content for item in service.list_records()], ["concurrent database"])
        self.assertEqual(len(service.repository.read_snapshot().records), 2)

    def test_history_is_retained_until_explicit_remove_and_current_fact_still_works(self):
        service = self.service()
        old = service.add("SQLite", fact_key="project.database", valid_from=NOW - timedelta(days=3))
        middle = service.revise_fact(old.id, "PostgreSQL", NOW - timedelta(days=1))
        current = service.revise_fact(middle.id, "current database", NOW)
        self.assertEqual(len(service.repository.read_snapshot().records), 3)
        self.assertEqual(service.list_records(), [current])
        self.assertTrue(service.remove(old.id))
        self.assertEqual(len(service.repository.read_snapshot().records), 2)
        self.assertEqual(service.list_records(), [current])
        self.assertEqual(service.list_records(MemoryQuery(as_of=NOW-timedelta(days=2))), [])
        self.assertEqual(service.get(middle.id).content, "PostgreSQL")
