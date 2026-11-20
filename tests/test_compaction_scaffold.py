"""C01—C04 外层配置/路由检查，不执行或伪装核心摘要算法。"""

from unittest.mock import Mock, patch

from context_lab.agents import ContextAwareAgent
from context_lab.config import AppConfig, ContextConfig
from context_lab.memory import MemoryRecord, MemoryService
from context_lab.models import ScriptedChatModel
from context_lab.tools.memory import MemoryTool
from context_lab.types import ContextBuildResult, ContextCompactionState, ContextPacket, ContextUsage, HistorySummaryRequest, Message
from tests.support import FakeBuilder, NOW, SandboxTestCase


class CompactionScaffoldTests(SandboxTestCase):
    def config(self, enabled=False):
        return AppConfig(state_dir=self.tmp / "state", user_id="alice",
                         context=ContextConfig(auto_summary_enabled=enabled))

    def test_default_thresholds_use_input_budget_and_are_separate_from_output_reserve(self):
        config = ContextConfig()
        self.assertFalse(config.auto_summary_enabled)
        self.assertEqual(config.input_budget, 2400)
        self.assertEqual(config.summary_trigger_tokens, 1920)
        self.assertEqual(config.summary_target_tokens, 1440)
        self.assertEqual(config.summary_keep_recent_messages, 4)
        self.assertEqual(config.summary_max_attempts_per_turn, 1)
        self.assertEqual(ContextConfig(max_tokens=3, reserve_ratio=0).summary_trigger_tokens, 3)

    def test_invalid_summary_configuration_is_rejected(self):
        cases = [{"auto_summary_enabled": 1}, {"summary_trigger_ratio": float("nan")},
                 {"summary_trigger_ratio": True}, {"summary_trigger_ratio": 1.1},
                 {"summary_target_ratio": 0}, {"summary_target_ratio": 0.8},
                 {"summary_keep_recent_messages": 0}, {"summary_keep_recent_messages": True},
                 {"summary_max_attempts_per_turn": 0}]
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                ContextConfig(**changes)

    def test_state_and_request_are_explicit_and_have_independent_history_identity(self):
        first, second = ContextCompactionState(), ContextCompactionState()
        self.assertNotEqual(first.history_id, second.history_id)
        self.assertEqual(first.covered_until, 0)
        self.assertIsNone(first.summary_id)
        first.covered_until = 2
        self.assertEqual(second.covered_until, 0)
        request = HistorySummaryRequest(first.history_id, "task-1", 0, 2,
                                       [Message("user", "before", NOW), Message("assistant", "result", NOW)], "prefix-hash")
        self.assertIsNone(request.previous_summary_id)
        self.assertIsNone(request.target_tokens)

    def test_enabled_constructor_does_not_read_files_or_call_a_model(self):
        model = ScriptedChatModel([])
        with patch("context_lab.storage.json_store.JsonFileStore.read", side_effect=AssertionError("构造不能读取")):
            agent = ContextAwareAgent(self.config(enabled=True), model)
        self.assertEqual(agent.compaction_state.covered_until, 0)
        self.assertIsNone(agent.last_context_usage)
        self.assertEqual(model.calls, [])
        self.assertFalse(agent.config.state_dir.exists())

    def test_disabled_entry_preserves_normal_builder_arguments_without_summary(self):
        builder = FakeBuilder()
        agent = ContextAwareAgent(self.config(), ScriptedChatModel([]), context_builder=builder)
        agent.history = [Message("user", "previous", NOW)]
        agent._build_with_summary = Mock(side_effect=AssertionError("关闭时不能摘要"))
        packets = [ContextPacket("note")]
        result = agent._build_context_for_turn("question", task_id="task-1", custom_packets=packets)
        self.assertIsInstance(result, ContextBuildResult)
        arguments = builder.calls[0][1]
        self.assertIs(arguments["conversation_history"], agent.history)
        self.assertIs(arguments["custom_packets"], packets)
        self.assertEqual(arguments["system_instructions"], agent.system_instructions)
        agent._build_with_summary.assert_not_called()

    def test_enabled_entry_delegates_to_compaction_core(self):
        builder = FakeBuilder()
        agent = ContextAwareAgent(self.config(enabled=True), ScriptedChatModel([]), context_builder=builder)
        expected = ContextBuildResult("prepared", [], [], 1)
        agent._build_with_summary = Mock(return_value=expected)
        self.assertIs(agent._build_context_for_turn("question", task_id="task-1"), expected)
        agent._build_with_summary.assert_called_once_with("question", task_id="task-1", custom_packets=None)
        self.assertEqual(builder.calls, [])

    def test_memory_tool_delegates_internal_summary_request_to_bound_service(self):
        config = self.config()
        service = MemoryService(config.memory_path, "alice")
        summary = MemoryRecord("summary", "checkpoint", user_id="alice", layer="summary")
        service.checkpoint_task = Mock(return_value=summary)
        tool = MemoryTool(config.memory_path, "alice", service=service)
        request = HistorySummaryRequest("history-1", "task-1", 0, 1, [Message("user", "old", NOW)], "hash")
        self.assertIs(tool.checkpoint_task(request), summary)
        service.checkpoint_task.assert_called_once_with(request)
        self.assertFalse(config.state_dir.exists())

    def test_clear_history_resets_coverage_identity_without_deleting_memory(self):
        agent = ContextAwareAgent(self.config(), ScriptedChatModel([]), context_builder=FakeBuilder())
        previous_id = agent.compaction_state.history_id
        agent.history.append(Message("user", "old", NOW))
        agent.compaction_state.covered_until = 1
        agent.compaction_state.summary_id = "summary-old"
        agent.last_context_usage = ContextUsage(100, 100, 80, False)
        agent.memory_tool.remove = Mock(side_effect=AssertionError("清历史不能删长期记忆"))
        agent.clear_history()
        self.assertEqual(agent.history, [])
        self.assertNotEqual(agent.compaction_state.history_id, previous_id)
        self.assertEqual(agent.compaction_state.covered_until, 0)
        self.assertIsNone(agent.compaction_state.summary_id)
        self.assertIsNone(agent.last_context_usage)
        agent.memory_tool.remove.assert_not_called()
