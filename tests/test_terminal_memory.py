import io
import json
from contextlib import redirect_stdout
from unittest.mock import patch

from context_lab.agents import CodebaseMaintainer
from context_lab.config import AppConfig, ContextConfig
from context_lab.models import ModelRequestError, ScriptedChatModel
from context_lab.terminal_ui import run_terminal
from context_lab.tools.sandbox import DEFAULT_PROJECT_TEST_COMMAND, DockerSandbox
from tests.support import SandboxTestCase


def route(target, action="analyze", preference=""):
    return json.dumps({"action": action, "target": target, "message": "", "user_preference": preference},
                      ensure_ascii=False)


class SummaryModel:
    def __init__(self, fail=False):
        self.fail = fail
        self.summaries = 0

    def generate(self, messages):
        instructions = messages[0].content
        if instructions.startswith("你负责理解当前终端"):
            return route("alpha")
        if instructions.startswith("整理仍在进行的任务"):
            self.summaries += 1
            if self.fail:
                raise ModelRequestError("timeout")
            return "阶段任务：排查连接问题。未完成：真实环境回归。"
        return "观察结论。" + "查验数据片段。" * 25


class TerminalMemoryTests(SandboxTestCase):
    def setUp(self):
        super().setUp()
        self.workspace = self.tmp / "workspace"
        (self.workspace / "alpha").mkdir(parents=True)
        (self.workspace / "beta").mkdir()

    def agent(self, model, user="alice", state="session", **options):
        context = ContextConfig(max_tokens=1800, reserve_ratio=0.1, auto_summary_enabled=True)
        config = AppConfig(workspace=self.workspace, state_dir=self.tmp / state, user_id=user, context=context)
        return CodebaseMaintainer(config, model, terminal_memory_dir=self.tmp / "memory", **options)

    def test_compaction_keeps_raw_history_and_can_be_reopened(self):
        model = SummaryModel()
        agent = self.agent(model)
        for step in range(5):
            reply = agent.converse(f"请求{step} 排查连接。" + "检查原始输出。" * 32)
        self.assertGreater(model.summaries, 0)
        self.assertEqual(len(agent.history), 10)
        self.assertIn("未完成：真实环境回归。", reply.context_result.context)
        self.assertLessEqual(reply.context_result.token_count, agent.config.context.input_budget)
        status = agent.session_status()["context_budget"]
        self.assertGreater(status["covered_messages"], 0)
        reopened = self.agent(SummaryModel())
        reopened.load_checkpoint()
        self.assertEqual(reopened.compaction_state.summary_id, status["summary_id"])
        self.assertEqual(len(reopened.history), 10)
        self.assertIn("未完成：真实环境回归。", reopened.converse("继续").context_result.context)

    def test_compaction_failure_uses_recent_context_within_the_budget(self):
        model = SummaryModel(fail=True)
        agent = self.agent(model)
        for step in range(5):
            reply = agent.converse(f"请求{step} 排查连接。" + "检查原始输出。" * 32)
        self.assertGreater(model.summaries, 0)
        self.assertEqual(agent.session_status()["context_budget"]["fallback"], "ModelRequestError")
        self.assertTrue(any("预算内的最近对话" in warning for warning in reply.context_result.warnings))
        self.assertLessEqual(reply.context_result.token_count, agent.config.context.input_budget)
        self.assertEqual(len(agent.history), 10)

    def test_resume_after_checkpoint_failure_does_not_duplicate_memory(self):
        preference = "我以后写 Python 测试优先用 unittest"
        agent = self.agent(ScriptedChatModel([route("alpha", preference=preference), "已记住"]))
        original_save = agent.save_checkpoint
        with patch.object(agent, "save_checkpoint", side_effect=[None, OSError("checkpoint write failed")]):
            original_save()
            with self.assertRaises(OSError):
                agent.converse(preference)
        resumed = self.agent(ScriptedChatModel([]))
        self.assertEqual(resumed.resume().answer, "已记住")
        stored = resumed.terminal_memory._preferences().repository.read_snapshot()
        self.assertEqual([record.content for record in stored.records], [preference])
        self.assertEqual(stored.revision, 1)

    def test_verified_progress_and_preferences_load_in_new_session_and_keep_scopes(self):
        alpha = self.workspace / "alpha"
        (alpha / "app.py").write_text("VALUE = 0\n", encoding="utf-8")
        sandbox = DockerSandbox("python:3.12-slim", DEFAULT_PROJECT_TEST_COMMAND)
        preference = "我以后写 Python 测试优先用 pytest"
        read = json.dumps({"action": "read_file", "arguments": {"relative_path": "alpha/app.py"}})
        submission = json.dumps({"action": "verify_edits", "arguments": {"edits": [
            {"path": "alpha/app.py", "old_text": "VALUE = 0", "new_text": "VALUE = 7"},
        ]}})
        first = self.agent(ScriptedChatModel([
            route("alpha", preference=preference), "记下了", route("alpha", "edit"), read, submission,
        ]), sandbox=sandbox)
        first.converse(preference)
        with (patch.object(sandbox, "ensure_ready"), patch.object(sandbox, "_run_test", return_value={
            "status": "passed", "exit_code": 0, "output": "one test passed",
        })):
            reply = first.converse("实现 ALPHA-ONLY 计数更新")
        self.assertTrue(reply.repair["applied_to"])
        self.assertTrue(any(event.metadata.get("location") == "app.py:1" for event in reply.trace))

        second_model = ScriptedChatModel([route("alpha"), "这是上次交付情况", route("beta"), "乙项目无进度"])
        second = self.agent(second_model)
        with patch("builtins.input", side_effect=["上次改到哪儿了", "/exit"]), redirect_stdout(io.StringIO()):
            self.assertEqual(run_terminal(second, new_session=True), 0)
        self.assertNotEqual(first.session_id, second.session_id)
        self.assertEqual(len(second.history), 2)
        self.assertIn("ALPHA-ONLY", second_model.calls[-1][-1].content)
        self.assertIn("实际验证", second_model.calls[-1][-1].content)
        self.assertIn(preference, second_model.calls[-1][-1].content)
        beta = second.converse("查看另一个项目").context_result.context
        self.assertNotIn("ALPHA-ONLY", beta)
        self.assertNotIn("这是上次交付情况", beta)
        self.assertIn(preference, beta)

        another = self.agent(ScriptedChatModel([route("alpha"), "新用户"]), user="bob", state="bob")
        isolated = another.converse("上次改到哪儿了").context_result.context
        self.assertNotIn("ALPHA-ONLY", isolated)
        self.assertNotIn(preference, isolated)
