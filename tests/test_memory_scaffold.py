"""已提供的 Memory 框架验收；通过不表示 M01—M07 核心已完成。"""

from dataclasses import replace
from datetime import datetime
from unittest.mock import Mock, patch

from context_lab.memory import (
    JsonMemoryRepository, MemoryCodec, MemoryConfig, MemoryContext, MemoryHit, MemoryQuery,
    MemoryRecord, MemoryService, MemorySnapshot, MemorySourceRef,
)
from context_lab.models import ScriptedChatModel
from context_lab.tools.memory import MemoryTool
from tests.memory_support import NOW, RecordingMemoryIndex, record
from tests.support import CharacterCounter, SandboxTestCase


class MemoryScaffoldTests(SandboxTestCase):
    def test_initialization_only_wires_dependencies(self):
        path = self.tmp / "state" / "memory.json"
        model, index, counter = ScriptedChatModel([]), RecordingMemoryIndex(), CharacterCounter()
        with patch("context_lab.storage.json_store.JsonFileStore.read", side_effect=AssertionError("不得读取")):
            service = MemoryService(path, "alice", embedding=index, summary_model=model, token_counter=counter)
            tool = MemoryTool(path, "alice", service=service)
        self.assertFalse(path.parent.exists())
        self.assertIs(tool.service, service)
        self.assertIs(service.retriever.vector, index)
        self.assertIs(service.hierarchy.model, model)
        self.assertEqual(index.applied, [])
        self.assertEqual(model.calls, [])
        self.assertEqual(counter.calls, [])

    def test_record_has_explicit_creation_and_update_times(self):
        item = record()
        self.assertIsInstance(item, MemoryRecord)
        self.assertEqual(item.created_at, NOW)
        self.assertEqual(item.updated_at, NOW)
        self.assertEqual(item.layer, "archive")
        self.assertIsNotNone(item.updated_at.utcoffset())

    def test_mutable_defaults_are_independent(self):
        left, right = MemoryRecord("a", "left"), MemoryRecord("b", "right")
        left.source_refs.append(MemorySourceRef("source", 1))
        left.metadata["x"] = 1
        self.assertEqual(right.source_refs, [])
        self.assertEqual(right.metadata, {})
        first, second = MemorySnapshot(), MemorySnapshot()
        first.records.append(left)
        self.assertEqual(second.records, [])

    def test_default_config_is_local_and_budget_is_explicit(self):
        config = MemoryConfig()
        self.assertEqual(config.search_mode, "lexical")
        self.assertEqual(sum((config.working_ratio, config.summary_ratio, config.facts_ratio, config.archive_ratio)), 1)
        for changes in ({"search_mode": "unknown"}, {"rrf_k": 0}, {"context_tokens": -1},
                        {"working_ratio": 1}, {"bm25_k1": float("nan")}, {"vector_weight": -1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(config, **changes)

    def test_cross_user_or_cross_file_injection_is_rejected(self):
        path = self.tmp / "memory.json"
        with self.assertRaises(ValueError):
            MemoryService(path, "alice", embedding=RecordingMemoryIndex("bob"))
        with self.assertRaises(ValueError):
            MemoryService(path, "alice", repository=JsonMemoryRepository(path, "bob"))
        service = MemoryService(path, "alice")
        with self.assertRaises(ValueError):
            MemoryTool(path, "bob", service=service)

    def test_search_wrapper_preserves_filter_objects_and_returns_records(self):
        service = MemoryService(self.tmp / "memory.json", "alice")
        item = record()
        service.search_hits = Mock(return_value=[MemoryHit(item, 1.0)])
        self.assertEqual(service.search("query", 3, mode="hybrid", as_of=NOW, layers=("summary",),
                                        min_importance=0), [item])
        args, kwargs = service.search_hits.call_args
        self.assertEqual(args, ("query", 3))
        self.assertEqual(kwargs["filters"], MemoryQuery(as_of=NOW, layers=("summary",), min_importance=0))

    def test_tool_delegates_and_parses_json_time(self):
        path = self.tmp / "memory.json"
        service = Mock(spec=MemoryService)
        service.user_id, service.path = "alice", path
        service.search.return_value = [record()]
        tool = MemoryTool(path, "alice", service=service)
        result = tool.run({"action": "search", "query": "query", "as_of": NOW.isoformat(), "layers": ["archive"]})
        self.assertTrue(result.success)
        service.search.assert_called_once_with("query", 5, as_of=NOW, layers=("archive",))
        with self.assertRaises(ValueError):
            tool.run({"action": "search", "query": "x", "user_id": "bob"})
        with self.assertRaises(ValueError):
            tool.run({"action": "search", "query": "x", "as_of": "2026-11-01T12:00:00"})

    def test_default_tool_constructs_the_same_service_and_add_delegates(self):
        path = self.tmp / "memories.json"
        default_tool = MemoryTool(path, "alice")
        self.assertIsInstance(default_tool.service, MemoryService)
        self.assertEqual(default_tool.service.user_id, "alice")
        self.assertFalse(path.exists())
        service = Mock(spec=MemoryService)
        service.user_id, service.path = "alice", path
        service.add.return_value = record(layer="working", task_id="task-1")
        tool = MemoryTool(path, "alice", service=service)
        self.assertEqual(tool.add("state", layer="working", task_id="task-1").layer, "working")
        service.add.assert_called_once()

    def test_missing_file_default_is_a_fresh_document(self):
        first = MemoryCodec.empty_document()
        first["records"].append({"id": "not-shared"})
        self.assertEqual(MemoryCodec.empty_document(),
                         {"schema_version": 1, "revision": 0, "records": [], "index_events": []})

    def test_optional_embedding_execution_is_explicit(self):
        index = RecordingMemoryIndex()
        service = MemoryService(self.tmp / "memory.json", "alice", embedding=index)
        self.assertEqual(index.process_calls, [])
        self.assertEqual(service.process_pending(3), 0)
        self.assertEqual(index.process_calls, [3])

    def test_context_render_exposes_identity_recording_and_valid_time(self):
        context = MemoryContext(working=[record(valid_from=NOW, task_id="task-1", layer="working")])
        rendered = context.render()
        self.assertIn("Current task", rendered)
        self.assertIn("Historical task summaries", rendered)
        self.assertIn("recorded=", rendered)
        self.assertIn("valid=[", rendered)
        self.assertIn("redis timeout", rendered)

    def test_advanced_action_routes_to_bound_service(self):
        path = self.tmp / "new.json"
        service = Mock(spec=MemoryService)
        service.user_id, service.path = "alice", path
        service.finish_task.return_value = record("summary", layer="summary")
        tool = MemoryTool(path, "alice", service=service)
        result = tool.run({"action": "finish_task", "task_id": "task-1"})
        service.finish_task.assert_called_once_with(task_id="task-1")
        self.assertEqual(result.data.layer, "summary")
        self.assertFalse(path.exists())
