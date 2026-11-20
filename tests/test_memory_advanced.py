"""M01—M07 行为验收：TODO 未完成时应失败，不跳过也不返回预设答案。"""

from copy import deepcopy
from datetime import timedelta
import json
from unittest.mock import patch

from context_lab.errors import ContextBudgetError, ModelNotConfigured
from context_lab.memory import (
    HybridMemoryRetriever, JsonMemoryRepository, LexicalMemoryRetriever, MemoryCodec,
    MemoryConfig, MemoryConflict, MemoryHierarchy, MemoryHit, MemoryIndexEvent, MemoryIndexPending,
    MemoryMutation, MemoryPolicy, MemoryQuery, MemoryService, MemorySnapshot, MemorySourceRef,
    MemorySummaryStale, MemoryWriter,
)
from context_lab.models import ScriptedChatModel
from context_lab.memory.writer import message_evidence_id
from context_lab.types import Message
from tests.memory_support import NOW, RecordingMemoryIndex, document, record, wire_record
from tests.support import CharacterCounter, SandboxTestCase


class CodecM01Tests(SandboxTestCase):
    def test_document_records_restore_without_changing_created_time(self):
        payload = document(wire_record())
        before = deepcopy(payload)
        snapshot = MemoryCodec().decode(payload)
        self.assertEqual(payload, before)
        item = snapshot.records[0]
        self.assertEqual(item.created_at, NOW)
        self.assertEqual(item.updated_at, NOW)
        self.assertEqual(item.layer, "archive")
        self.assertEqual(item.version, 1)
        self.assertEqual(snapshot.revision, 0)

    def test_round_trip_preserves_effective_time_and_source_versions(self):
        item = record(layer="summary", source_refs=[MemorySourceRef("raw-1", 2)], valid_from=NOW,
                      valid_to=NOW + timedelta(days=1))
        snapshot = MemorySnapshot(3, [item], [MemoryIndexEvent("event-1", "alice", item.id, "upsert", item.version, NOW)])
        payload = MemoryCodec().encode(snapshot)
        encoded = json.loads(json.dumps(payload, allow_nan=False))
        self.assertEqual(MemoryCodec().decode(encoded), snapshot)
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["revision"], 3)

    def test_repository_read_filters_user_without_rewriting_file(self):
        path = self.tmp / "memories.json"
        path.write_text(json.dumps(document(wire_record("a", user_id="alice"), wire_record("b", user_id="bob"))), encoding="utf-8")
        before = path.read_bytes()
        snapshot = JsonMemoryRepository(path, "alice").read_snapshot()
        self.assertEqual([item.id for item in snapshot.records], ["a"])
        self.assertEqual(path.read_bytes(), before)

    def test_missing_file_is_empty_without_creating_directories(self):
        path = self.tmp / "absent" / "memory.json"
        self.assertEqual(JsonMemoryRepository(path, "alice").read_snapshot(), MemorySnapshot())
        self.assertFalse(path.parent.exists())

    def test_list_format_is_not_accepted(self):
        with self.assertRaises(ValueError):
            MemoryCodec().decode([wire_record()])


class PersistenceM02Tests(SandboxTestCase):
    def service(self, user="alice", **kwargs):
        return MemoryService(self.tmp / "memory.json", user, clock=lambda: NOW, **kwargs)

    def test_add_saves_record_and_event_then_recovers_after_restart(self):
        service = self.service()
        item = service.add("new memory", layer="working", task_id="task-1")
        snapshot = self.service().repository.read_snapshot()
        self.assertEqual(snapshot.records, [item])
        self.assertEqual(item.created_at, NOW)
        self.assertEqual(item.updated_at, NOW)
        self.assertEqual(snapshot.index_events[0].memory_id, item.id)
        self.assertEqual(snapshot.index_events[0].operation, "upsert")

    def test_two_users_keep_each_others_records_and_events(self):
        alice, bob = self.service(), self.service("bob")
        a, b = alice.add("alice"), bob.add("bob")
        self.assertEqual([item.id for item in self.service().repository.read_snapshot().records], [a.id])
        self.assertEqual([item.id for item in self.service("bob").repository.read_snapshot().records], [b.id])
        self.assertFalse(bob.remove(a.id))

    def test_update_keeps_created_time_and_accepts_zero_importance(self):
        service = self.service()
        first = service.add("before", importance=0.8)
        service.clock = lambda: NOW + timedelta(hours=1)
        updated = service.update(first.id, content="after", importance=0)
        self.assertEqual(updated.created_at, first.created_at)
        self.assertEqual(updated.updated_at, NOW + timedelta(hours=1))
        self.assertEqual(updated.version, first.version + 1)
        self.assertEqual(updated.importance, 0)

    def test_stale_snapshot_is_rejected(self):
        repository = self.service().repository
        first, stale = repository.read_snapshot(), repository.read_snapshot()
        first.records.append(record())
        repository.commit(first)
        before = repository.path.read_bytes()
        with self.assertRaises(MemoryConflict):
            repository.commit(stale)
        self.assertEqual(repository.path.read_bytes(), before)

    def test_write_failure_does_not_report_success_or_leave_partial_memory(self):
        service = self.service()
        with patch.object(service.repository.store, "write", side_effect=OSError("test write failure")):
            with self.assertRaises(OSError):
                service.add("cannot save")
        self.assertFalse(service.path.exists())


class TemporalM03Tests(SandboxTestCase):
    def test_valid_interval_is_closed_on_left_open_on_right(self):
        item = record(valid_from=NOW, valid_to=NOW + timedelta(hours=1))
        policy = MemoryPolicy()
        self.assertEqual(policy.select([item], "alice", MemoryQuery(as_of=NOW), NOW), [item])
        self.assertEqual(policy.select([item], "alice", MemoryQuery(as_of=item.valid_to), NOW), [])
        self.assertEqual(policy.select([item], "alice", MemoryQuery(as_of=NOW - timedelta(seconds=1)), NOW), [])

    def test_recording_time_does_not_replace_effective_time(self):
        item = record(created_at=NOW, valid_from=NOW - timedelta(days=10))
        past = NOW - timedelta(days=5)
        self.assertEqual(MemoryPolicy().select([item], "alice", MemoryQuery(as_of=past), NOW), [item])

    def test_users_layers_stale_summaries_and_zero_importance(self):
        records = [record("a", importance=0), record("b", user_id="bob"),
                   record("s", layer="summary", summary_state="stale")]
        selected = MemoryPolicy().select(records, "alice", MemoryQuery(max_importance=0), NOW)
        self.assertEqual([item.id for item in selected], ["a"])

    def test_revision_keeps_old_fact_and_switches_at_effective_time(self):
        service = MemoryService(self.tmp / "memory.json", "alice", clock=lambda: NOW)
        old = service.add("database is SQLite", memory_type="semantic", fact_key="project.database",
                          valid_from=NOW - timedelta(days=1))
        boundary = NOW + timedelta(hours=1)
        new = service.revise_fact(old.id, "database is PostgreSQL", boundary)
        self.assertNotEqual(old.id, new.id)
        self.assertEqual(new.supersedes_id, old.id)
        before = service.list_records(MemoryQuery(as_of=NOW))
        after = service.list_records(MemoryQuery(as_of=boundary))
        self.assertEqual([item.content for item in before], ["database is SQLite"])
        self.assertEqual([item.content for item in after], ["database is PostgreSQL"])
        self.assertEqual(new.created_at, NOW)


class RetrievalM04Tests(SandboxTestCase):
    def test_deterministic_tokenization_preserves_identifiers(self):
        self.assertEqual(LexicalMemoryRetriever().tokenize("你好 get_cached HTTP_429!"),
                         ["你好", "get_cached", "http_429"])

    def test_bm25_returns_matching_records_without_unrelated_fillers(self):
        target, unrelated = record("a", "get_cached vector cache"), record("b", "weekly planning")
        hits = LexicalMemoryRetriever().search("get_cached", [unrelated, target], 5)
        self.assertEqual([hit.memory.id for hit in hits], ["a"])
        self.assertGreater(hits[0].score, 0)

    def test_rrf_fuses_ranks_instead_of_incompatible_raw_scores(self):
        a, b, c = record("a"), record("b"), record("c")
        retriever = HybridMemoryRetriever(LexicalMemoryRetriever())
        hits = retriever.fuse([MemoryHit(a, 100), MemoryHit(b, 90)],
                              [MemoryHit(b, 0.9), MemoryHit(c, 0.8)], 3)
        self.assertEqual([hit.memory.id for hit in hits], ["b", "a", "c"])

    def test_hybrid_search_limits_results_after_fusing_both_rankings(self):
        a = record("a", "redis redis")
        b = record("b", "redis cache")
        c = record("c", "weekly planning")
        index = RecordingMemoryIndex()
        index.hits = [MemoryHit(c, 0.9), MemoryHit(b, 0.8)]
        retriever = HybridMemoryRetriever(LexicalMemoryRetriever(), index)

        hits = retriever.search("redis", [a, b, c], limit=1, mode="hybrid")

        self.assertEqual([hit.memory.id for hit in hits], ["b"])
        self.assertEqual(hits[0].ranks, {"lexical": 2, "vector": 2})

    def test_filtering_precedes_final_limit(self):
        service = MemoryService(self.tmp / "memory.json", "alice", clock=lambda: NOW)
        for i in range(6):
            service.add("redis redis redis timeout", valid_to=NOW, metadata={"old": i})
        valid = service.add("redis timeout")
        hits = service.search("redis", limit=1, as_of=NOW)
        self.assertEqual([item.id for item in hits], [valid.id])

    def test_lexical_search_does_not_call_embedding(self):
        index = RecordingMemoryIndex()
        service = MemoryService(self.tmp / "memory.json", "alice", embedding=index, clock=lambda: NOW)
        item = service.add("get_cached Vector")
        self.assertEqual(service.search("get_cached", mode="lexical")[0].id, item.id)
        self.assertEqual(index.search_calls, [])


class IndexingM05Tests(SandboxTestCase):
    def service(self, index):
        return MemoryService(self.tmp / "memory.json", "alice", embedding=index, clock=lambda: NOW)

    def test_repeated_updates_coalesce_to_latest_record(self):
        index = RecordingMemoryIndex()
        service = self.service(index)
        item = service.add("before")
        updated = service.update(item.id, "after")
        report = service.flush_index_events()
        self.assertEqual(report.applied, 1)
        self.assertEqual(index.applied, [(item.id, updated)])
        self.assertEqual(service.repository.read_snapshot().index_events, [])
        self.assertEqual(index.process_calls, [])

    def test_removed_record_is_not_revived_by_old_upsert_event(self):
        index = RecordingMemoryIndex()
        service = self.service(index)
        item = service.add("temporary")
        service.remove(item.id)
        service.flush_index_events()
        self.assertEqual(index.applied, [(item.id, None)])

    def test_failed_index_sync_preserves_business_and_pending_events(self):
        index = RecordingMemoryIndex()
        service = self.service(index)
        item = service.add("durable fact")
        index.apply_error = RuntimeError("test database unavailable")
        report = service.flush_index_events()
        self.assertEqual(service.get(item.id).content, "durable fact")
        self.assertEqual(report.applied, 0)
        self.assertEqual(report.pending, 1)
        self.assertEqual(report.errors, {item.id: "index_sync_failed"})
        self.assertTrue(service.repository.read_snapshot().index_events)

    def test_vector_search_with_pending_events_is_explicitly_blocked(self):
        index = RecordingMemoryIndex()
        service = self.service(index)
        service.add("not synced")
        for mode in ("vector", "hybrid"):
            with self.subTest(mode=mode), self.assertRaises(MemoryIndexPending):
                service.search("query", mode=mode)
        self.assertEqual(index.search_calls, [])

    def test_rebuild_appends_events_for_existing_records_and_preserves_pending_deletes(self):
        index = RecordingMemoryIndex()
        service = self.service(index)
        removed = service.add("already removed")
        service.remove(removed.id)
        snapshot = service.repository.read_snapshot()
        snapshot.records.extend([
            record("historical", "previous database", fact_key="project.database", valid_to=NOW, version=2),
            record("fresh", layer="summary", summary_state="fresh"),
            record("stale", layer="summary", summary_state="stale", version=3),
        ])
        before = service.repository.commit(snapshot)
        other = MemoryService(service.path, "bob", clock=lambda: NOW)
        other.add("other user's memory")
        other_before = other.repository.read_snapshot()

        self.assertEqual(service.rebuild_index(), 3)

        after = service.repository.read_snapshot()
        self.assertEqual(after.records, before.records)
        self.assertEqual(after.index_events[:len(before.index_events)], before.index_events)
        added_events = after.index_events[len(before.index_events):]
        self.assertEqual(
            [(event.memory_id, event.operation, event.version) for event in added_events],
            [("historical", "upsert", 2), ("fresh", "upsert", 1), ("stale", "delete", 3)],
        )
        other_after = other.repository.read_snapshot()
        self.assertEqual(other_after.records, other_before.records)
        self.assertEqual(other_after.index_events, other_before.index_events)
        self.assertEqual(index.applied, [])
        self.assertEqual(index.process_calls, [])

    def test_rebuild_retries_and_uses_the_current_record_version(self):
        service = self.service(RecordingMemoryIndex())
        item = service.add("before")
        initial_revision = service.repository.read_snapshot().revision
        concurrent = self.service(RecordingMemoryIndex())
        commit = service.repository.commit

        def commit_after_concurrent_update(snapshot):
            if snapshot.revision == initial_revision:
                concurrent.update(item.id, "after")
            return commit(snapshot)

        with patch.object(service.repository, "commit", side_effect=commit_after_concurrent_update) as commits:
            self.assertEqual(service.rebuild_index(), 1)

        self.assertEqual(commits.call_count, 2)
        snapshot = service.repository.read_snapshot()
        self.assertEqual(snapshot.records[0].content, "after")
        self.assertEqual([event.version for event in snapshot.index_events], [1, 2, 2])


class HierarchyM06Tests(SandboxTestCase):
    def service(self, model):
        return MemoryService(self.tmp / "memory.json", "alice", summary_model=model,
                             token_counter=CharacterCounter(), clock=lambda: NOW)

    def test_summarize_keeps_model_score_source_versions_and_time_range(self):
        model = ScriptedChatModel([json.dumps({"summary": "本地测试验证了缓存修复", "importance": 0.8})])
        first_time = NOW - timedelta(hours=2)
        last_time = NOW - timedelta(hours=1)
        records = [
            record("first", "排查缓存超时", task_id="task-1", version=2, created_at=first_time),
            record("last", "本地用例通过", task_id="task-1", version=3, created_at=last_time),
        ]

        summary = MemoryHierarchy(model=model).summarize(records, "task-1", "alice", NOW)

        self.assertEqual((summary.layer, summary.user_id, summary.task_id), ("summary", "alice", "task-1"))
        self.assertEqual((summary.content, summary.importance), ("本地测试验证了缓存修复", 0.8))
        self.assertEqual(summary.source_refs, [MemorySourceRef("first", 2), MemorySourceRef("last", 3)])
        self.assertEqual((summary.created_at, summary.updated_at), (NOW, NOW))
        self.assertEqual((summary.valid_from, summary.valid_to), (None, None))
        self.assertEqual(summary.metadata["first_record_at"], first_time.isoformat())
        self.assertEqual(summary.metadata["last_record_at"], last_time.isoformat())
        self.assertEqual(len(model.calls), 1)
        self.assertEqual([message.role for message in model.calls[0]], ["system", "user"])
        self.assertIn("本地用例通过", model.calls[0][1].content)

    def test_summarize_rejects_other_users_and_tasks_before_model_call(self):
        model = ScriptedChatModel([])
        hierarchy = MemoryHierarchy(model=model)
        for source in (record(task_id="task-2"), record(task_id="task-1", user_id="bob")):
            with self.subTest(user=source.user_id, task=source.task_id), self.assertRaises(ValueError):
                hierarchy.summarize([source], "task-1", "alice", NOW)
        self.assertEqual(model.calls, [])

    def test_summarize_rejects_malformed_model_replies(self):
        source = record(task_id="task-1")
        replies = [
            "plain text",
            "[]",
            json.dumps({"summary": " ", "importance": 0.5}),
            json.dumps({"summary": "summary", "importance": "0.8"}),
            json.dumps({"summary": "summary", "importance": 0.9}),
        ]
        for reply in replies:
            with self.subTest(reply=reply), self.assertRaises(ValueError):
                hierarchy = MemoryHierarchy(model=ScriptedChatModel([reply]))
                hierarchy.summarize([source], "task-1", "alice", NOW)

    def test_finish_task_archives_raw_and_keeps_source_versions(self):
        model = ScriptedChatModel([json.dumps({"summary": "修复缓存，本地测试通过", "importance": 0.5})])
        service = self.service(model)
        first = service.add("目标：修复缓存", layer="working", task_id="task-1")
        second = service.add("结果：测试通过", layer="working", task_id="task-1")
        unrelated = service.add("另一任务的工作记录", layer="working", task_id="task-2")
        before = service.repository.read_snapshot()
        summary = service.finish_task("task-1")
        self.assertEqual(summary.layer, "summary")
        self.assertEqual({ref.memory_id for ref in summary.source_refs}, {first.id, second.id})
        self.assertTrue(all(service.get(ref.memory_id).layer == "archive" for ref in summary.source_refs))
        self.assertTrue(all(service.get(ref.memory_id).version == ref.version for ref in summary.source_refs))
        self.assertEqual(service.get(unrelated.id), unrelated)
        saved = service.repository.read_snapshot()
        self.assertEqual(saved.index_events[:len(before.index_events)], before.index_events)
        new_events = saved.index_events[len(before.index_events):]
        self.assertEqual(
            [(event.memory_id, event.version, event.operation) for event in new_events],
            [(first.id, 2, "upsert"), (second.id, 2, "upsert"), (summary.id, 1, "upsert")],
        )
        self.assertEqual(service.finish_task("task-1"), summary)
        self.assertEqual(service.repository.read_snapshot(), saved)
        self.assertEqual(len(model.calls), 1)

    def test_finish_task_reuses_draft_on_conflict_only_for_unchanged_sources(self):
        for source_changed in (False, True):
            with self.subTest(source_changed=source_changed):
                path = self.tmp / f"conflict-{source_changed}.json"
                model = ScriptedChatModel([
                    json.dumps({"summary": "initial summary", "importance": 0.5}),
                    json.dumps({"summary": "updated summary", "importance": 0.8}),
                ])
                service = MemoryService(path, "alice", summary_model=model, clock=lambda: NOW)
                concurrent = MemoryService(path, "alice", clock=lambda: NOW)
                source = service.add("original evidence", layer="working", task_id="task-1")
                initial_revision = service.repository.read_snapshot().revision
                commit = service.repository.commit

                def commit_after_concurrent_change(snapshot):
                    if snapshot.revision == initial_revision:
                        if source_changed:
                            concurrent.update(source.id, "updated evidence")
                        else:
                            concurrent.add("unrelated evidence", layer="working", task_id="task-2")
                    return commit(snapshot)

                with patch.object(service.repository, "commit", side_effect=commit_after_concurrent_change) as commits:
                    summary = service.finish_task("task-1")

                self.assertEqual(commits.call_count, 2)
                self.assertEqual(len(model.calls), 2 if source_changed else 1)
                self.assertEqual(summary.content, "updated summary" if source_changed else "initial summary")
                self.assertEqual(summary.source_refs, [MemorySourceRef(source.id, 3 if source_changed else 2)])
                self.assertEqual(service.get(source.id).version, summary.source_refs[0].version)

    def test_summary_failure_does_not_save_half_finished_task(self):
        service = self.service(ScriptedChatModel([]))
        item = service.add("state", layer="working", task_id="task-1")
        before = service.path.read_bytes()
        with self.assertRaises(ModelNotConfigured):
            service.finish_task("task-1")
        self.assertEqual(service.path.read_bytes(), before)
        self.assertEqual(service.get(item.id).layer, "working")

    def test_expansion_rejects_changed_source(self):
        source = record("raw", version=2)
        summary = record("summary", layer="summary", source_refs=[MemorySourceRef("raw", 1)])
        with self.assertRaises(MemorySummaryStale):
            MemoryHierarchy().expand(summary, [source])

    def test_expansion_preserves_reference_order_and_copies_sources(self):
        first = record("first", version=2, metadata={"tags": ["cache"]})
        last = record("last")
        other_user = record("first", content="other user's evidence", user_id="bob", version=2)
        summary = record("summary", layer="summary", source_refs=[
            MemorySourceRef("first", 2), MemorySourceRef("last", 1),
        ])

        sources = MemoryHierarchy().expand(summary, [last, first, other_user])

        self.assertEqual(sources, [first, last])
        sources[0].metadata["tags"].append("changed")
        self.assertEqual(first.metadata["tags"], ["cache"])

    def test_expansion_rejects_missing_sources_and_stale_summaries(self):
        source = record("raw")
        summary = record("summary", layer="summary", source_refs=[MemorySourceRef("raw", 1)])
        hierarchy = MemoryHierarchy()

        with self.assertRaises(MemorySummaryStale):
            hierarchy.expand(summary, [])
        summary.summary_state = "stale"
        with self.assertRaises(MemorySummaryStale):
            hierarchy.expand(summary, [source])

    def test_expand_summary_uses_one_snapshot_and_propagates_stale_state(self):
        model = ScriptedChatModel([json.dumps({"summary": "task summary", "importance": 0.5})])
        service = self.service(model)
        source = service.add("original evidence", layer="working", task_id="task-1")
        summary = service.finish_task("task-1")

        read_snapshot = service.repository.read_snapshot
        with patch.object(service.repository, "read_snapshot", wraps=read_snapshot) as read:
            evidence = service.expand_summary(summary.id)

        read.assert_called_once_with()
        self.assertEqual(evidence, [service.get(source.id)])
        with self.assertRaisesRegex(ValueError, "找不到摘要"):
            service.expand_summary("missing")
        service.update(source.id, "corrected evidence")
        with self.assertRaises(MemorySummaryStale):
            service.expand_summary(summary.id)
        self.assertEqual(len(model.calls), 1)

    def test_changed_source_invalidates_summary_then_explicit_rebuild_repairs_it(self):
        model = ScriptedChatModel([
            json.dumps({"summary": "first summary", "importance": 0.5}),
            json.dumps({"summary": "updated summary", "importance": 0.8}),
        ])
        service = self.service(model)
        raw = service.add("first evidence", layer="working", task_id="task-1")
        summary = service.finish_task("task-1")
        current = service.update(raw.id, "corrected evidence")
        stale = service.get(summary.id)
        self.assertEqual(stale.summary_state, "stale")
        rebuilt = service.rebuild_summary(summary.id)
        self.assertEqual(rebuilt.id, summary.id)
        self.assertEqual(rebuilt.created_at, summary.created_at)
        self.assertEqual(rebuilt.summary_state, "fresh")
        self.assertEqual(rebuilt.source_refs, [MemorySourceRef(raw.id, current.version)])
        self.assertEqual(rebuilt.version, stale.version + 1)
        self.assertEqual((rebuilt.content, rebuilt.importance), ("updated summary", 0.8))
        self.assertEqual(service.expand_summary(summary.id), [current])
        saved = service.repository.read_snapshot()
        event = saved.index_events[-1]
        self.assertEqual((event.memory_id, event.version, event.operation), (summary.id, rebuilt.version, "upsert"))
        self.assertEqual(service.rebuild_summary(summary.id), rebuilt)
        self.assertEqual(service.repository.read_snapshot(), saved)
        self.assertEqual(len(model.calls), 2)

    def test_rebuild_summary_regenerates_when_source_changes_during_commit(self):
        model = ScriptedChatModel([
            json.dumps({"summary": "initial summary", "importance": 0.5}),
            json.dumps({"summary": "first attempt", "importance": 0.5}),
            json.dumps({"summary": "latest evidence summarized", "importance": 0.8}),
        ])
        service = self.service(model)
        source = service.add("original evidence", layer="working", task_id="task-1")
        summary = service.finish_task("task-1")
        service.update(source.id, "first correction")
        concurrent = MemoryService(service.path, "alice", clock=lambda: NOW)
        initial_revision = service.repository.read_snapshot().revision
        commit = service.repository.commit

        def commit_after_concurrent_update(snapshot):
            if snapshot.revision == initial_revision:
                concurrent.update(source.id, "latest correction")
            return commit(snapshot)

        with patch.object(service.repository, "commit", side_effect=commit_after_concurrent_update) as commits:
            rebuilt = service.rebuild_summary(summary.id)

        self.assertEqual(commits.call_count, 2)
        self.assertEqual(len(model.calls), 3)
        self.assertIn("latest correction", model.calls[-1][1].content)
        self.assertEqual(rebuilt.content, "latest evidence summarized")
        self.assertEqual(service.expand_summary(rebuilt.id), [concurrent.get(source.id)])

    def test_context_keeps_current_detail_within_rendered_budget(self):
        current = record("working", "current blocker", layer="working", task_id="task-1")
        old = record("old", "old logs " * 100, layer="archive", task_id="old-task")
        counter = CharacterCounter()
        context = MemoryHierarchy(token_counter=counter).build_context([current, old], "task-1", "alice", 400)
        self.assertIn(current.id, [item.id for item in context.working])
        self.assertLessEqual(counter.count(context.render()), 400)
        self.assertEqual(context.token_count, counter.count(context.render()))

    def test_context_keeps_working_over_its_share_but_rejects_total_overflow(self):
        current = record("working", "x" * 100, layer="working", task_id="task-1")
        hierarchy = MemoryHierarchy(token_counter=CharacterCounter())

        context = hierarchy.build_context([current], "task-1", "alice", 300)
        self.assertEqual(context.working, [current])
        with self.assertRaises(ContextBudgetError):
            hierarchy.build_context([current], "task-1", "alice", context.token_count - 1)

    def test_context_skips_oversized_candidates_and_duplicate_records(self):
        current = record("working", layer="working", task_id="task-1")
        other_user = record("working", layer="working", task_id="task-1", user_id="bob")
        other_task = record("other-task", layer="working", task_id="task-2")
        long_summary = record("long-summary", "x" * 500, layer="summary")
        short_summary = record("short-summary", layer="summary")
        stale_summary = record("stale-summary", layer="summary", summary_state="stale")
        fact = record("fact", fact_key="project.database")
        evidence = record("evidence", "x" * 300)
        records = [other_user, current, current, other_task, long_summary, short_summary, stale_summary,
                   fact, fact, evidence]
        counter = CharacterCounter()

        context = MemoryHierarchy(token_counter=counter).build_context(records, "task-1", "alice", 1200)

        self.assertEqual(context.working, [current])
        self.assertEqual(context.summaries, [short_summary])
        self.assertEqual(context.facts, [fact])
        self.assertEqual(context.archive, [])
        self.assertEqual(len(context.warnings), 1)
        self.assertEqual(context.token_count, counter.count(context.render()))

    def test_prepare_context_preserves_working_and_filters_ranked_summary_evidence(self):
        model = ScriptedChatModel([json.dumps({"summary": "redis timeout", "importance": 0.8})])
        service = self.service(model)
        expired = service.add("expired evidence", layer="working", task_id="old", valid_to=NOW)
        sources = [service.add(f"evidence {i}", layer="working", task_id="old") for i in range(3)]
        summary = service.finish_task("old")
        working = service.add("network already checked", layer="working", task_id="current")
        service.add("other task state", layer="working", task_id="other")
        former_fact = service.add("redis former host", fact_key="redis.host", valid_to=NOW)
        current_fact = service.add("redis current host", fact_key="redis.host", valid_from=NOW)
        snapshot = service.repository.read_snapshot()
        empty_summary = record("a-empty", "redis timeout", layer="summary", task_id="no-refs")
        snapshot.records.append(empty_summary)
        service.repository.commit(snapshot)

        read_snapshot = service.repository.read_snapshot
        with patch.object(service.repository, "read_snapshot", wraps=read_snapshot) as read:
            context = service.prepare_context("redis", "current", max_tokens=10000)

        read.assert_called_once_with()
        self.assertEqual(context.working, [working])
        self.assertEqual([item.id for item in context.summaries], [empty_summary.id, summary.id])
        self.assertEqual(context.facts, [current_fact])
        self.assertEqual([item.id for item in context.archive], [item.id for item in sources[:2]])
        self.assertEqual(context.token_count, service.hierarchy.token_counter.count(context.render()))

        earlier = service.prepare_context("redis", "current", max_tokens=10000, as_of=NOW - timedelta(seconds=1))
        self.assertEqual(earlier.facts, [former_fact])
        self.assertEqual([item.id for item in earlier.archive], [expired.id, sources[0].id])
        unmatched = service.prepare_context("unmatched", "current")
        self.assertEqual(unmatched.working, [working])
        self.assertEqual(unmatched.archive, [])
        self.assertEqual(len(model.calls), 1)

    def test_prepare_context_checks_pending_events_before_vector_retrieval(self):
        for mode in ("vector", "hybrid"):
            with self.subTest(mode=mode):
                index = RecordingMemoryIndex()
                service = MemoryService(
                    self.tmp / f"{mode}.json", "alice", config=MemoryConfig(search_mode=mode),
                    embedding=index, token_counter=CharacterCounter(), clock=lambda: NOW,
                )
                working = service.add("network already checked", layer="working", task_id="current")
                fact = service.add("redis host", fact_key="redis.host")
                index.hits = [MemoryHit(fact, 0.9)]

                with self.assertRaises(MemoryIndexPending):
                    service.prepare_context("redis", "current")
                self.assertEqual(service.prepare_context(" ", "current").working, [working])
                self.assertEqual(index.search_calls, [])

                service.flush_index_events()
                context = service.prepare_context("redis", "current")
                self.assertEqual(context.working, [working])
                self.assertEqual(context.facts, [fact])
                self.assertEqual(len(index.search_calls), 1)


class WriterM07Tests(SandboxTestCase):
    def test_model_output_is_proposal_and_does_not_mutate_existing(self):
        existing = [record("old", "database PostgreSQL", memory_type="semantic", fact_key="project.database")]
        before = deepcopy(existing)
        model = ScriptedChatModel([json.dumps([{
            "operation": "noop", "content": "database PostgreSQL", "target_id": "old",
            "expected_version": 1, "fact_key": "project.database", "reason": "already exists",
            "evidence_ids": ["old"],
        }])])
        proposals = MemoryWriter(model).propose([Message("user", "database PostgreSQL")], existing)
        self.assertIsInstance(proposals[0], MemoryMutation)
        self.assertEqual(proposals[0].operation, "noop")
        self.assertEqual(existing, before)

    def test_noop_does_not_create_another_record(self):
        service = MemoryService(self.tmp / "memory.json", "alice", clock=lambda: NOW)
        item = service.add("database PostgreSQL", memory_type="semantic", fact_key="project.database")
        service.apply_mutations([MemoryMutation("noop", item.content, item.id, item.version,
                                               fact_key=item.fact_key, evidence_ids=[item.id])])
        self.assertEqual(len(service.repository.read_snapshot().records), 1)

    def test_proposal_and_atomic_apply_keep_fact_history(self):
        service = MemoryService(self.tmp / "memory.json", "alice", clock=lambda: NOW)
        old = service.add("database SQLite", fact_key="project.database")
        effective = NOW - timedelta(days=1)
        message = Message("user", f"项目数据库从 {effective.isoformat()} 开始使用 PostgreSQL", NOW)
        service.writer.model = ScriptedChatModel([json.dumps([{
            "operation": "revise", "content": "database PostgreSQL", "target_id": old.id,
            "expected_version": old.version, "fact_key": old.fact_key,
            "effective_from": effective.isoformat(), "evidence_ids": [message_evidence_id(message)],
        }])])
        before = service.repository.read_snapshot()

        proposals = service.propose([message])
        self.assertEqual(service.repository.read_snapshot(), before)
        new, = service.apply_mutations(proposals)

        saved = service.repository.read_snapshot()
        self.assertEqual(saved.revision, before.revision + 1)
        self.assertEqual(service.get(old.id).valid_to, effective)
        self.assertEqual(new.valid_from, effective)
        self.assertEqual(new.supersedes_id, old.id)
        self.assertEqual(new.metadata["write_evidence"][0]["content"], message.content)
        self.assertEqual([item.content for item in service.list_records(MemoryQuery(as_of=effective))],
                         ["database PostgreSQL"])
        self.assertEqual([item.content for item in service.list_records(
            MemoryQuery(as_of=effective - timedelta(seconds=1))
        )], ["database SQLite"])

    def test_invalid_mutation_leaves_the_entire_batch_unwritten(self):
        service = MemoryService(self.tmp / "memory.json", "alice", clock=lambda: NOW)
        ordinary = service.add("initial evidence")
        old = service.add("database SQLite", fact_key="project.database", valid_from=NOW)
        before = service.repository.read_snapshot()
        mutations = [
            MemoryMutation("update", "corrected evidence", ordinary.id, ordinary.version,
                           evidence_ids=[ordinary.id]),
            MemoryMutation("revise", "database PostgreSQL", old.id, old.version,
                           effective_from=NOW - timedelta(days=1), evidence_ids=[old.id]),
        ]
        with self.assertRaises(ValueError):
            service.apply_mutations(mutations)
        self.assertEqual(service.repository.read_snapshot(), before)
