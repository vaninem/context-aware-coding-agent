"""已提供的脚手架检查；初始版本应通过，不代表核心练习完成。"""

from context_lab.agents import CodebaseMaintainer, ContextAwareAgent, ProjectAssistant
from context_lab.config import AppConfig, ContextConfig
from context_lab.context import ContextBuilder
from context_lab.errors import ModelNotConfigured
from context_lab.models import HTTPChatModel, ScriptedChatModel
from context_lab.storage.json_store import JsonFileStore
from context_lab.tools.base import BaseTool
from context_lab.tools.memory import MemoryTool
from context_lab.tools.notes import NoteTool
from context_lab.tools.rag import RAGTool
from context_lab.tools.terminal import TerminalTool
from context_lab.types import ContextPacket, Message
from tests.support import CharacterCounter, FakeBuilder, FakeNotes, FakeRetriever, FakeTerminal, SandboxTestCase


class ScaffoldTests(SandboxTestCase):
    def config(self):
        workspace = self.tmp / "workspace"
        workspace.mkdir(exist_ok=True)
        return AppConfig(workspace=workspace, state_dir=self.tmp / "state")

    def test_configuration_budget_and_validation(self):
        self.assertEqual(ContextConfig(max_tokens=100, reserve_ratio=0.25).input_budget, 75)
        for values in [{"max_tokens": 0}, {"history_limit": -1}, {"reserve_ratio": 1},
                       {"relevance_weight": 0.6, "recency_weight": 0.6}]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                ContextConfig(**values)

    def test_all_tools_initialize_without_running_exercises(self):
        counter = CharacterCounter()
        tools = [
            NoteTool(self.tmp / "notes"), MemoryTool(self.tmp / "memories.json", "alice"),
            RAGTool(self.tmp / "knowledge", token_counter=counter), TerminalTool(self.tmp),
        ]
        self.assertTrue(all(isinstance(tool, BaseTool) for tool in tools))
        self.assertEqual(tools[0].index, {})
        self.assertEqual(tools[1].service.user_id, "alice")
        self.assertEqual(tools[1].user_id, "alice")
        self.assertEqual(tools[2].chunks, [])
        self.assertEqual(counter.calls, [])
        for tool in tools:
            with self.subTest(tool=type(tool).__name__), self.assertRaises(ValueError):
                tool.run({"action": "not-an-action"})

    def test_builder_keeps_injected_dependencies_without_counting(self):
        counter, memory, rag = CharacterCounter(), FakeRetriever(), FakeRetriever()
        config = ContextConfig()
        builder = ContextBuilder(config, memory, rag, counter)
        self.assertIs(builder.config, config)
        self.assertIs(builder.memory_tool, memory)
        self.assertIs(builder.rag_tool, rag)
        self.assertIs(builder.token_counter, counter)
        self.assertEqual(counter.calls, [])
        self.assertEqual(memory.calls, [])
        self.assertEqual(rag.calls, [])

    def test_agent_class_hierarchy_and_dependency_wiring(self):
        config = self.config()
        model, builder = ScriptedChatModel([]), FakeBuilder()
        memory, rag, notes, terminal = FakeRetriever(), FakeRetriever(), FakeNotes(), FakeTerminal()
        agent = CodebaseMaintainer(config, model, memory_tool=memory, rag_tool=rag,
                                   context_builder=builder, note_tool=notes, terminal_tool=terminal)
        self.assertIsInstance(agent, ProjectAssistant)
        self.assertIsInstance(agent, ContextAwareAgent)
        for name, value in [("config", config), ("model", model), ("context_builder", builder),
                            ("memory_tool", memory), ("rag_tool", rag), ("note_tool", notes),
                            ("terminal_tool", terminal)]:
            self.assertIs(getattr(agent, name), value)
        self.assertEqual(agent.history, [])
        self.assertEqual(agent.trace, [])
        self.assertTrue(agent.session_id)
        self.assertIsInstance(agent.stats, dict)
        self.assertEqual(model.calls, [])

    def test_json_store_round_trip_and_corruption_are_explicit(self):
        path = self.tmp / "state" / "data.json"
        store = JsonFileStore(path)
        self.assertEqual(store.read(default={"initial": True}), {"initial": True})
        store.write({"中文": [1, 2], "ok": True})
        self.assertEqual(JsonFileStore(path).read(), {"中文": [1, 2], "ok": True})
        path.write_text("{broken", encoding="utf-8")
        with self.assertRaises(ValueError):
            store.read()

    def test_scripted_model_records_snapshot_and_reports_exhaustion(self):
        model = ScriptedChatModel(["ready"])
        messages = [Message("user", "original")]
        self.assertEqual(model.generate(messages), "ready")
        messages[0].content = "changed"
        self.assertEqual(model.calls[0][0].content, "original")
        with self.assertRaises(ModelNotConfigured):
            model.generate(messages)

    def test_http_model_constructor_only_saves_explicit_configuration(self):
        model = HTTPChatModel("example-model", "https://invalid.example/v1/", api_key_env="UNSET_LAB_KEY", timeout_seconds=2)
        self.assertEqual(model.model_name, "example-model")
        self.assertEqual(model.base_url, "https://invalid.example/v1")
        self.assertEqual(model.calls, 0)

    def test_shared_types_use_aware_times_and_independent_mutable_defaults(self):
        left, right = ContextPacket("left"), ContextPacket("right")
        self.assertIsNotNone(left.timestamp.utcoffset())
        left.metadata["source"] = "one"
        self.assertNotIn("source", right.metadata)
