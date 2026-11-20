"""E09：替换上下文、笔记与取证工具，隔离验证编排。"""

import json

from context_lab.agents import CodebaseMaintainer, ContextAwareAgent, ProjectAssistant
from context_lab.config import AppConfig
from context_lab.errors import ExercisePending, ModelNotConfigured
from context_lab.memory import MemoryService
from context_lab.models import ScriptedChatModel
from context_lab.tools.memory import MemoryTool
from context_lab.tools.rag import RAGTool
from context_lab.types import AgentReply, Message, TraceEvent
from tests.support import FakeBuilder, FakeNotes, FakeTerminal, NOW, SandboxTestCase


class AgentTests(SandboxTestCase):
    def config(self):
        workspace = self.tmp / "workspace"
        workspace.mkdir(exist_ok=True)
        return AppConfig(workspace=workspace, state_dir=self.tmp / "state", user_id="alice")

    def test_base_agent_uses_builder_model_and_records_history(self):
        model, builder = ScriptedChatModel(["answer-one", "answer-two"]), FakeBuilder()
        agent = ContextAwareAgent(self.config(), model, context_builder=builder)
        first = agent.run("question-one")
        self.assertIsInstance(first, AgentReply)
        self.assertEqual(first.answer, "answer-one")
        self.assertEqual(first.context_result.context, "验收上下文：question-one")
        self.assertTrue(any("验收上下文" in m.content for m in model.calls[0]))
        self.assertEqual([m.role for m in agent.history[-2:]], ["user", "assistant"])
        self.assertEqual([m.content for m in agent.history[-2:]], ["question-one", "answer-one"])
        agent.run("question-two")
        history_passed = builder.calls[1][1].get("conversation_history")
        self.assertIsNotNone(history_passed)
        self.assertTrue(any(m.content == "answer-one" for m in history_passed))
        self.assertTrue(first.trace)
        self.assertTrue(all(isinstance(event, TraceEvent) for event in first.trace))

    def test_pending_context_is_not_turned_into_a_fake_success(self):
        model = ScriptedChatModel(["should not be called"])
        pending = ExercisePending("E06", "pending")
        builder = FakeBuilder(pending)
        agent = ContextAwareAgent(self.config(), model, context_builder=builder)
        with self.assertRaises(ExercisePending) as caught:
            agent.run("question")
        self.assertIs(caught.exception, pending, "必须传播上下文依赖的原始异常")
        self.assertEqual(len(builder.calls), 1)
        self.assertEqual(model.calls, [])

    def test_base_agent_can_opt_in_to_evidence_based_memory_write(self):
        class WriterModel:
            def generate(self, messages):
                evidence, = json.loads(messages[1].content)["messages"]
                return json.dumps([{
                    "operation": "add", "content": evidence["content"], "fact_key": "project.database",
                    "evidence_ids": [evidence["id"]],
                }])

        config = self.config()
        config.auto_memory_write_enabled = True
        service = MemoryService(config.memory_path, "alice", summary_model=WriterModel())
        tool = MemoryTool(config.memory_path, "alice", service=service)
        agent = ContextAwareAgent(config, ScriptedChatModel(["noted"]), memory_tool=tool, context_builder=FakeBuilder())

        reply = agent.run("项目使用 PostgreSQL")

        item, = service.list_records()
        self.assertEqual(item.content, "项目使用 PostgreSQL")
        self.assertEqual(item.fact_key, "project.database")
        self.assertEqual(reply.trace[-1].stage, "memory_write")

    def test_project_notes_are_delegated_and_keep_source_ids(self):
        notes = FakeNotes()
        agent = ProjectAssistant(self.config(), ScriptedChatModel([]), note_tool=notes, context_builder=FakeBuilder())
        note = agent.record_note("排查记录", "redis timeout", "runbook", ["cache"])
        packets = agent._notes_to_packets("timeout")
        self.assertTrue(packets)
        self.assertTrue(all(p.metadata.get("type") == "note" for p in packets))
        self.assertIn(note.id, str([p.metadata for p in packets]))
        self.assertIn("redis timeout", "\n".join(p.content for p in packets))

    def test_maintainer_exploration_uses_read_only_evidence(self):
        terminal, notes, builder = FakeTerminal(), FakeNotes(), FakeBuilder()
        model = ScriptedChatModel(["exploration", "analysis", "plan"])
        agent = CodebaseMaintainer(self.config(), model, terminal_tool=terminal, note_tool=notes, context_builder=builder)
        for call in [agent.explore, lambda: agent.analyze_issue("health check timeout"), lambda: agent.plan_task("add retries")]:
            self.assertIsInstance(call(), AgentReply)
        self.assertEqual(len(model.calls), 3)
        self.assertTrue(terminal.calls, "项目助手必须实际调用只读取证接口")

    def test_checkpoint_restores_history_and_session_in_a_new_instance(self):
        config = self.config()
        first = CodebaseMaintainer(config, ScriptedChatModel([]), terminal_tool=FakeTerminal(), note_tool=FakeNotes(), context_builder=FakeBuilder())
        first.history = [Message("user", "remember this", NOW), Message("assistant", "saved", NOW)]
        first.stats["tool_calls"] = 7
        session_id = first.session_id
        saved_path = first.save_checkpoint()
        self.assertEqual(saved_path, config.checkpoint_path)
        self.assertTrue(saved_path.is_file())
        second = CodebaseMaintainer(config, ScriptedChatModel([]), terminal_tool=FakeTerminal(), note_tool=FakeNotes(), context_builder=FakeBuilder())
        restored = second.load_checkpoint()
        self.assertIsInstance(restored, dict)
        self.assertEqual(second.session_id, session_id)
        self.assertEqual([(m.role, m.content) for m in second.history], [("user", "remember this"), ("assistant", "saved")])
        self.assertTrue(all(m.timestamp.utcoffset() is not None for m in second.history))
        self.assertEqual(second.get_stats()["tool_calls"], 7)

    def test_langgraph_chooses_tools_and_stops_after_the_limit(self):
        model = ScriptedChatModel([
            json.dumps({"action": "search", "arguments": {"query": f"needle-{i}"}}) for i in range(4)
        ] + ['{"answer":"依据 app.py:1 得出结论"}'])
        terminal = FakeTerminal()
        agent = CodebaseMaintainer(self.config(), model, terminal_tool=terminal, note_tool=FakeNotes())

        reply = agent.analyze_issue("health check")

        self.assertEqual([call[1] for call in terminal.calls], [f"needle-{i}" for i in range(4)])
        self.assertIn("def health_check():", model.calls[-1][1].content)
        self.assertEqual(reply.answer, "依据 app.py:1 得出结论")
        self.assertEqual(agent.get_stats()["model_calls"], 5)
        self.assertEqual(agent.get_stats()["tool_calls"], 4)

    def test_langgraph_resumes_saved_evidence_without_repeating_the_tool(self):
        config, terminal = self.config(), FakeTerminal()
        first_model = ScriptedChatModel(['{"action":"read_file","arguments":{"relative_path":"app.py"}}'])
        first = CodebaseMaintainer(config, first_model, terminal_tool=terminal, note_tool=FakeNotes())
        with self.assertRaises(ModelNotConfigured):
            first.analyze_issue("health_check")
        self.assertEqual(first.history, [])

        second_model = ScriptedChatModel(["恢复后完成"])
        second = CodebaseMaintainer(config, second_model, terminal_tool=terminal, note_tool=FakeNotes())
        reply = second.resume()

        self.assertEqual(reply.answer, "恢复后完成")
        self.assertEqual(second.session_id, first.session_id)
        self.assertEqual(terminal.calls, [("read", "app.py")])
        self.assertIn("def health_check():", second_model.calls[0][1].content)
        self.assertEqual([message.role for message in second.history], ["user", "assistant"])
        self.assertEqual(second.get_stats()["tool_calls"], 1)
        with self.assertRaisesRegex(ValueError, "没有待恢复"):
            second.resume()

    def test_langgraph_reuses_the_rag_tool_without_a_terminal_dependency(self):
        config = self.config()
        (config.workspace / "main.py").write_text("def needle(): return 'marker'\n", encoding="utf-8")
        rag = RAGTool(config.workspace)
        rag.add_document(config.workspace / "main.py")
        model = ScriptedChatModel([
            '{"action":"rag_search","arguments":{"query":"needle"}}', '{"answer":"看到定义"}',
        ])
        agent = CodebaseMaintainer(config, model, rag_tool=rag)

        reply = agent.analyze_issue("needle")

        self.assertIn("main.py", model.calls[-1][1].content)
        self.assertIn("marker", model.calls[-1][1].content)
        self.assertEqual(reply.answer, "看到定义")
        self.assertTrue(any(event.stage == "note_unavailable" for event in reply.trace))
