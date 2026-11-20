"""C01—C04 阶段摘要行为验收。"""

from copy import deepcopy
from datetime import timezone
import hashlib
import json
from unittest.mock import Mock

from context_lab.agents import ContextAwareAgent
from context_lab.config import AppConfig, ContextConfig
from context_lab.context import ContextBuilder
from context_lab.errors import ModelNotConfigured
from context_lab.memory import MemoryHierarchy, MemoryRecord, MemoryService, MemorySourceRef
from context_lab.models import ScriptedChatModel
from context_lab.tools.memory import MemoryTool
from context_lab.types import ContextBuildResult, ContextCompactionState, ContextPacket, ContextUsage, HistorySummaryRequest, Message
from tests.memory_support import record
from tests.support import CharacterCounter, NOW, SandboxTestCase


def prefix_hash(messages):
    """验收采用的规范指纹：按序保存角色、原文和 UTC 时间的紧凑 JSON。"""
    payload = [{"role": message.role, "content": message.content,
                "timestamp": message.timestamp.astimezone(timezone.utc).isoformat()}
               for message in messages]
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class CompactionTests(SandboxTestCase):
    def history(self):
        return [Message("user", "U"*1000, NOW), Message("assistant", "A"*1000, NOW),
                Message("user", "recent-1", NOW), Message("assistant", "recent-2", NOW),
                Message("user", "recent-3", NOW), Message("assistant", "recent-4", NOW)]

    def test_c01_checks_full_history_before_history_limit_and_keeps_recent_interactions(self):
        history = self.history()
        before = deepcopy(history)
        builder = ContextBuilder(ContextConfig(history_limit=1), token_counter=CharacterCounter())
        usage = builder.inspect_usage("next", history, "rules", state=ContextCompactionState())
        self.assertGreaterEqual(usage.estimated_tokens, 2000)
        self.assertEqual(usage.trigger_tokens, 1920)
        self.assertTrue(usage.should_summarize)
        self.assertEqual((usage.history_start, usage.history_end), (0, 2))
        self.assertEqual(history, before)

    def test_c01_replaces_covered_prefix_instead_of_double_counting_or_resummarizing(self):
        state = ContextCompactionState(task_id="task-1", covered_until=2,
                                       prefix_hash=prefix_hash(self.history()[:2]), summary_id="summary-1")
        builder = ContextBuilder(token_counter=CharacterCounter())
        summary = ContextPacket("previous checkpoint", metadata={"type": "progress_summary"})
        usage = builder.inspect_usage("next", self.history(), "rules", state=state, summary_packet=summary)
        self.assertLess(usage.estimated_tokens, 1920)
        self.assertFalse(usage.should_summarize)
        self.assertEqual((usage.history_start, usage.history_end), (2, 2))

    def test_c01_does_not_claim_system_overflow_is_compressible_history(self):
        builder = ContextBuilder(token_counter=CharacterCounter())
        usage = builder.inspect_usage("question", [], "S"*3000)
        self.assertGreater(usage.estimated_tokens, usage.input_budget)
        self.assertFalse(usage.should_summarize)
        self.assertEqual(usage.history_start, usage.history_end)

    def test_c02_progress_summary_keeps_evidence_and_unresolved_work_without_mutating_working(self):
        source = record("step-1", content="排查记录：连接池达到上限，正在定位等待请求，修复尚未验证。"*20,
                        layer="working", task_id="task-1")
        before = deepcopy(source)
        model = ScriptedChatModel(["目标：修复超时；已定位连接池；尚未验证修复。"])
        summary = MemoryHierarchy(model=model, token_counter=CharacterCounter()).summarize_progress(
            [source], "task-1", "alice", NOW,
        )
        self.assertEqual(summary.layer, "summary")
        self.assertEqual(summary.metadata["summary_kind"], "progress")
        self.assertEqual(summary.task_id, "task-1")
        self.assertEqual(summary.source_refs, [MemorySourceRef(source.id, source.version)])
        self.assertIn("尚未", summary.content)
        self.assertEqual(source, before)
        self.assertEqual(len(model.calls), 1)

    def test_c03_persists_messages_summary_and_outbox_once_and_reuses_identical_request(self):
        model = ScriptedChatModel(["阶段进展：定位超时原因，等待验证修复。"])
        service = MemoryService(self.tmp / "memory.json", "alice", summary_model=model,
                                token_counter=CharacterCounter(), clock=lambda: NOW)
        messages = [Message("user", "查超时原因。"*20, NOW),
                    Message("assistant", "已定位连接池，尚未验证。"*20, NOW)]
        request = HistorySummaryRequest("history-1", "task-1", 0, 2, messages, prefix_hash(messages))
        before = deepcopy(request)
        summary = service.checkpoint_task(request)
        saved = service.repository.read_snapshot()
        self.assertEqual(len(summary.source_refs), 2)
        self.assertEqual(summary.metadata["covered_until"], 2)
        self.assertEqual(summary.metadata["history_id"], request.history_id)
        self.assertEqual(summary.metadata["summary_kind"], "progress")
        for ref in summary.source_refs:
            source = service.get(ref.memory_id)
            self.assertEqual(source.layer, "archive")
            self.assertEqual(source.version, ref.version)
            self.assertEqual(source.task_id, "task-1")
        self.assertEqual(len(saved.records), 3)
        self.assertEqual(len(saved.index_events), 3)
        self.assertEqual(service.checkpoint_task(request).id, summary.id)
        self.assertEqual(service.repository.read_snapshot().revision, saved.revision)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(request, before)

    def test_c03_model_failure_does_not_save_half_a_summary_or_sources(self):
        path = self.tmp / "missing" / "memory.json"
        service = MemoryService(path, "alice", summary_model=ScriptedChatModel([]), clock=lambda: NOW)
        messages = [Message("user", "old question", NOW), Message("assistant", "old reply", NOW)]
        request = HistorySummaryRequest("history-1", "task-1", 0, 2, messages, prefix_hash(messages))
        with self.assertRaises(ModelNotConfigured):
            service.checkpoint_task(request)
        self.assertFalse(path.exists())

    def test_c03_extends_checkpoint_and_preserves_original_references(self):
        model = ScriptedChatModel(["初次定位，等待验证。", "已定位连接池，仍待回归验证。"])
        service = MemoryService(self.tmp / "memory.json", "alice", summary_model=model,
                                token_counter=CharacterCounter(), clock=lambda: NOW)
        messages = [Message(role, "排查结果待验证。" * 80, NOW)
                    for role in ("user", "assistant", "user", "assistant")]
        first = service.checkpoint_task(HistorySummaryRequest(
            "history-1", "task-1", 0, 2, messages[:2], prefix_hash(messages[:2]),
        ))
        request = HistorySummaryRequest(
            "history-1", "task-1", 2, 4, messages[2:], prefix_hash(messages), previous_summary_id=first.id,
        )
        second = service.checkpoint_task(request)
        saved = service.repository.read_snapshot()
        self.assertEqual(second.source_refs[:2], first.source_refs)
        self.assertEqual(len(second.source_refs), 4)
        self.assertEqual(service.get(first.id).summary_state, "stale")
        self.assertTrue(any(event.memory_id == first.id and event.operation == "delete" for event in saved.index_events))
        self.assertEqual(service.checkpoint_task(request), second)
        self.assertEqual(service.repository.read_snapshot(), saved)
        self.assertEqual(len(model.calls), 2)

    def test_c04_below_threshold_builds_context_without_requesting_a_summary(self):
        config = AppConfig(state_dir=self.tmp / "state", user_id="alice",
                           context=ContextConfig(auto_summary_enabled=True))
        builder = ContextBuilder(config.context, token_counter=CharacterCounter())
        expected = ContextBuildResult("prepared", [], [], 8)
        builder.inspect_usage = Mock(return_value=ContextUsage(100, 2400, 1920, False))
        builder.build = Mock(return_value=expected)
        agent = ContextAwareAgent(config, ScriptedChatModel([]), context_builder=builder)
        agent.memory_tool.checkpoint_task = Mock(side_effect=AssertionError("未触发不能摘要"))
        self.assertIs(agent._build_context_for_turn("next", task_id="task-1"), expected)
        agent.memory_tool.checkpoint_task.assert_not_called()

    def test_c04_updates_coverage_after_success_rebuilds_view_and_retains_raw_history(self):
        config = AppConfig(state_dir=self.tmp / "state", user_id="alice",
                           context=ContextConfig(auto_summary_enabled=True))
        builder = ContextBuilder(config.context, token_counter=CharacterCounter())
        builder.inspect_usage = Mock(side_effect=[ContextUsage(2100, 2400, 1920, True, 0, 2),
                                                  ContextUsage(500, 2400, 1920, False, 2, 2)])
        expected = ContextBuildResult("prepared", [], [], 8)
        builder.build = Mock(return_value=expected)
        service = MemoryService(config.memory_path, "alice")
        tool = MemoryTool(config.memory_path, "alice", service=service)
        summary = MemoryRecord("summary-1", "阶段进展：等待验证。", user_id="alice", layer="summary",
                               task_id="task-1", metadata={"summary_kind": "progress"})
        tool.checkpoint_task = Mock(return_value=summary)
        model = ScriptedChatModel([])
        agent = ContextAwareAgent(config, model, memory_tool=tool, context_builder=builder)
        agent.history = self.history()
        before = deepcopy(agent.history)
        self.assertIs(agent._build_context_for_turn("next", task_id="task-1"), expected)
        self.assertEqual(tool.checkpoint_task.call_count, 1)
        self.assertEqual(agent.compaction_state.covered_until, 2)
        self.assertEqual(agent.compaction_state.summary_id, summary.id)
        self.assertTrue(agent.compaction_state.prefix_hash)
        self.assertEqual(builder.inspect_usage.call_count, 2)
        self.assertEqual(agent.history, before)
        arguments = builder.build.call_args.kwargs
        sent_history = arguments.get("conversation_history", [])
        packets = arguments.get("custom_packets") or []
        sent_text = "\n".join(item.content for item in [*sent_history, *packets])
        self.assertIn(summary.content, sent_text)
        self.assertNotIn(before[0].content, sent_text)
        self.assertTrue(all(message.content in sent_text for message in before[2:]))
        self.assertEqual(model.calls, [])

    def test_run_saves_checkpoint_before_answer_and_reuses_it_after_answer_failure(self):
        config = AppConfig(state_dir=self.tmp / "state", user_id="alice",
                           context=ContextConfig(auto_summary_enabled=True, history_limit=1))
        counter = CharacterCounter()
        summary_model = ScriptedChatModel(["oldterm 阶段摘要，等待验证。"])
        service = MemoryService(config.memory_path, "alice", summary_model=summary_model, token_counter=counter)
        tool = MemoryTool(config.memory_path, "alice", service=service)
        builder = ContextBuilder(config.context, memory_tool=tool, token_counter=counter)
        agent = ContextAwareAgent(config, ScriptedChatModel([]), memory_tool=tool, context_builder=builder)
        agent.history = self.history()
        agent.history[0].content = "oldterm " + agent.history[0].content
        before = deepcopy(agent.history)

        with self.assertRaises(ModelNotConfigured):
            agent.run("oldterm", task_id="task-1")

        saved = service.repository.read_snapshot()
        self.assertEqual(agent.compaction_state.covered_until, 2)
        self.assertEqual(agent.history, before)
        agent.model = ScriptedChatModel(["reply"])
        answer = agent.run("oldterm")
        self.assertEqual(answer.answer, "reply")
        self.assertEqual(service.repository.read_snapshot(), saved)
        self.assertEqual(len(summary_model.calls), 1)
        self.assertIn("等待验证", answer.context_result.context)
        self.assertNotIn(before[0].content, answer.context_result.context)
        self.assertTrue(all(message.content in answer.context_result.context for message in before[2:]))
        self.assertEqual(agent.history[:-2], before)
        self.assertLessEqual(answer.context_result.token_count, config.context.input_budget)
