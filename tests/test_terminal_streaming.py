import io
import json
from contextlib import redirect_stdout
from unittest.mock import patch

from context_lab.agents import CodebaseMaintainer
from context_lab.agents.workflow import apply_verified_patch
from context_lab.config import AppConfig
from context_lab.models import HTTPChatModel, ModelRequestError, ScriptedChatModel
from context_lab.terminal_ui import run_terminal
from context_lab.tools.sandbox import DEFAULT_PROJECT_TEST_COMMAND, DockerSandbox
from context_lab.types import Message
from tests.support import SandboxTestCase


def sse(choice: dict) -> bytes:
    return f"data: {json.dumps({'choices': [choice]}, ensure_ascii=False)}\r\n\r\n".encode()


class StreamingModel(ScriptedChatModel):
    def __init__(self, replies, stream):
        super().__init__(replies)
        self.stream = stream

    def generate_stream(self, messages):
        yield from self.stream(messages)


class TerminalStreamingTests(SandboxTestCase):
    def test_http_deltas_arrive_before_end_and_ignore_reasoning_and_other_candidates(self):
        data = (sse({"index": 0, "delta": {"reasoning_content": "内部文本"}})
                + sse({"index": 1, "delta": {"content": "其他候选"}})
                + sse({"index": 0, "delta": {"content": "第一段"}})
                + sse({"index": 0, "delta": {"content": "第二段"}})
                + sse({"index": 0, "delta": {}, "finish_reason": "stop"})
                + b"data: [DONE]\r\n\r\n")
        response = io.BytesIO(data)
        model = HTTPChatModel("example", "https://unused.example", api_key="fake", json_output=True)
        with patch("context_lab.models.urlopen", return_value=response) as open_response:
            stream = model.generate_stream([Message("user", "你好")])
            self.assertEqual(next(stream), "第一段")
            self.assertLess(response.tell(), len(data))
            self.assertEqual(list(stream), ["第二段"])
        request = open_response.call_args.args[0]
        self.assertTrue(json.loads(request.data)["stream"])
        self.assertNotIn("response_format", json.loads(request.data))
        self.assertEqual(request.get_header("Accept"), "text/event-stream")
        self.assertTrue(response.closed)

    def test_truncated_http_stream_is_recoverable(self):
        model = HTTPChatModel("example", "https://unused.example", api_key="fake")
        response = io.BytesIO(sse({"delta": {"content": "部分答复"}}))
        with patch("context_lab.models.urlopen", return_value=response):
            stream = model.generate_stream([Message("user", "你好")])
            self.assertEqual(next(stream), "部分答复")
            with self.assertRaises(ModelRequestError) as error:
                next(stream)
        self.assertTrue(error.exception.retryable)
        self.assertTrue(response.closed)

    def test_terminal_streams_only_the_final_reply_once(self):
        workspace = self.tmp / "repo"
        workspace.mkdir()
        (workspace / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
        config = AppConfig(workspace=workspace, state_dir=self.tmp / "state")
        shown = io.StringIO()

        def stream(messages):
            self.assertIn("常量在 app.py", messages[-1].content)
            yield "结果在"
            self.assertIn("结果在", shown.getvalue())
            yield " app.py。"

        model = StreamingModel([
            '{"action":"analyze","target":".","message":""}',
            '{"action":"read_file","arguments":{"relative_path":"app.py"}}',
            '{"answer":"常量在 app.py"}',
        ], stream)
        agent = CodebaseMaintainer(config, model)
        with patch("builtins.input", side_effect=["看看常量在哪里", "/exit"]), redirect_stdout(shown):
            self.assertEqual(run_terminal(agent), 0)
        self.assertEqual(shown.getvalue().count("结果在 app.py。"), 1)
        self.assertNotIn('"action"', shown.getvalue())
        self.assertEqual(agent.history[-1].content, "结果在 app.py。")

    def test_resume_after_interrupt_does_not_repeat_patch_delivery_or_tests(self):
        workspace = self.tmp / "repo"
        workspace.mkdir()
        source = workspace / "app.py"
        source.write_text("VALUE = 1\n", encoding="utf-8")
        config = AppConfig(workspace=workspace, state_dir=self.tmp / "state")
        sandbox = DockerSandbox("python:3.12-slim", DEFAULT_PROJECT_TEST_COMMAND)
        closed = []

        def interrupted(messages):
            try:
                yield "尚未输出完的正文"
                raise KeyboardInterrupt
            finally:
                closed.append(True)

        model = StreamingModel([
            '{"action":"edit","target":".","message":""}',
            '{"action":"verify_edits","arguments":{"edits":[{"path":"app.py","old_text":"VALUE = 1","new_text":"VALUE = 2"}]}}',
        ], interrupted)
        agent = CodebaseMaintainer(config, model, sandbox=sandbox)
        passed = {"status": "passed", "exit_code": 0, "output": "test passed"}
        shown = io.StringIO()
        with (patch.object(sandbox, "ensure_ready"), patch.object(sandbox, "_run_test", return_value=passed) as run,
              patch("context_lab.agents.workflow.apply_verified_patch", wraps=apply_verified_patch) as apply,
              redirect_stdout(shown)):
            with patch("builtins.input", side_effect=["把 VALUE 改成 2", "/status", "/exit"]):
                self.assertEqual(run_terminal(agent), 0)
            self.assertEqual(agent.session_status()["next"], ("render_answer",))
            self.assertEqual(agent.history, [])
            self.assertEqual(source.read_text(), "VALUE = 2\n")
            self.assertEqual(closed, [True])

            def recovered(messages):
                self.assertIn("修改已通过测试并应用", messages[-1].content)
                yield "恢复后的完整正文。"

            reopened = CodebaseMaintainer(config, StreamingModel([], recovered))
            with patch("builtins.input", side_effect=["/resume", "/exit"]):
                self.assertEqual(run_terminal(reopened), 0)
            self.assertEqual(run.call_count, 2)
            apply.assert_called_once()
        self.assertIn("当前阶段 生成最终答复", shown.getvalue())
        self.assertEqual(len(reopened.history), 2)
        self.assertEqual(reopened.history[-1].content, "恢复后的完整正文。")
