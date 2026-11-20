import gzip
import io
import json
import shutil
import subprocess
from pathlib import Path
from contextlib import redirect_stdout
from unittest.mock import patch

from context_lab.agents import CodebaseMaintainer
from context_lab.config import AppConfig
from context_lab.errors import ModelNotConfigured
from context_lab.models import HTTPChatModel, ScriptedChatModel
from context_lab.rag.evaluation import RerankComparison
from context_lab.rag.repoqa import evaluate_repoqa
from context_lab.terminal_ui import display_progress, run_terminal
from context_lab.tools.delivery import base_commit, deliver
from context_lab.tools.sandbox import DockerSandbox, tree_digest
from context_lab.tools.symbols import PythonSymbols
from context_lab.types import Message
from tests.support import FakeNotes, SandboxTestCase


SOURCE = "def normalize(value):\n    return value.lower()\n"


def candidate(method: str) -> str:
    return ("--- a/app.py\n+++ b/app.py\n@@ -1,2 +1,2 @@\n def normalize(value):\n"
            f"-    return value.lower()\n+    return value.{method}().lower()\n")


def verify_patch(diff: str) -> str:
    return json.dumps({"action": "verify_patch", "arguments": {"patch": diff}})


class RepairTests(SandboxTestCase):
    def setUp(self):
        super().setUp()
        self.workspace = self.tmp / "workspace"
        self.workspace.mkdir()
        self.source = self.workspace / "app.py"
        self.source.write_text(SOURCE, encoding="utf-8")
        self.config = AppConfig(workspace=self.workspace, state_dir=self.workspace / "state")
        self.sandbox = DockerSandbox("python:3.12-slim", ("python", "-m", "unittest"))

    def maintainer(self, replies):
        return CodebaseMaintainer(self.config, ScriptedChatModel(replies), sandbox=self.sandbox,
                                  note_tool=FakeNotes())

    def test_terminal_carries_previous_evidence_and_exports_tested_patch(self):
        answer = "源码只小写化，未清理周围空格。"
        read = json.dumps({"action": "read_file", "arguments": {"relative_path": "app.py"}})
        analyze = json.dumps({"action": "analyze", "target": ".", "message": ""})
        edit = json.dumps({"action": "edit", "target": ".", "message": ""})
        model = ScriptedChatModel([analyze, read, answer, edit, verify_patch(candidate("strip")), analyze, "第三轮回答"])
        agent = CodebaseMaintainer(self.config, model, sandbox=self.sandbox, note_tool=FakeNotes(),
                                  carry_context=True, progress=display_progress)
        baseline = {"status": "passed", "exit_code": 0, "output": "existing tests passed"}
        passed = {"status": "passed", "patch": candidate("strip"), "exit_code": 0, "output": "OK"}
        shown = io.StringIO()
        with (
            patch("builtins.input", side_effect=["先检查入口", "按刚才结论修改", "还有什么", "/status", "/exit"]),
            patch.object(self.sandbox, "ensure_ready"),
            patch.object(self.sandbox, "baseline", return_value={"reproduction": baseline}),
            patch.object(self.sandbox, "verify", return_value=passed),
            redirect_stdout(shown),
        ):
            self.assertEqual(run_terminal(agent), 0)

        self.assertIn(answer, model.calls[3][-1].content)
        self.assertIn(answer, model.calls[4][-1].content)
        self.assertIn(SOURCE.strip(), model.calls[4][-1].content)
        self.assertIn("最近验证为第 1 次提交，结果 passed", shown.getvalue())
        self.assertIn("本轮上下文约", shown.getvalue())
        self.assertIn("正在验证候选补丁", shown.getvalue())
        artifact, = (self.config.state_dir / "artifacts").iterdir()
        self.assertEqual((artifact / "repair.patch").read_text(), candidate("strip"))
        report = json.loads((artifact / "report.json").read_text())
        self.assertTrue(any(event["metadata"].get("ran_tests") for event in report["trace"]))
        self.assertEqual(self.source.read_text(), SOURCE.replace(".lower()", ".strip().lower()"))

    def test_project_generation_retries_and_remembers_new_directory(self):
        route = json.dumps({"action": "create", "target": "todo", "message": ""})
        draft = json.dumps({"files": {"todo/todo.py": "VALUE = 1\n", "todo/tests/test_todo.py": "# test\n",
                                      "todo/README.md": "# TODO\n"},
                            "remove": [], "verify": True})
        revision = json.dumps({"files": {"todo.py": "VALUE = 2\n"}, "remove": [], "verify": True})
        model = ScriptedChatModel([route, draft, revision])
        agent = CodebaseMaintainer(self.config, model, sandbox=self.sandbox, note_tool=FakeNotes())
        target = self.workspace / "todo"
        observed = []

        def verify(staged):
            self.assertFalse(target.exists())
            self.assertTrue((staged / "tests/test_todo.py").is_file())
            text = (staged / "todo.py").read_text()
            observed.append(text)
            passed = text == "VALUE = 2\n"
            return {"status": "passed" if passed else "failed", "exit_code": int(not passed), "output": "actual != expected"}

        with patch.object(agent.project_sandbox, "verify_project", side_effect=verify):
            reply = agent.converse("新建一个 todo 项目")
        self.assertEqual(reply.project["status"], "generated")
        self.assertEqual(observed, ["VALUE = 1\n", "VALUE = 2\n"])
        self.assertIn("actual != expected", model.calls[-1][-1].content)
        self.assertEqual((target / "todo.py").read_text(), "VALUE = 2\n")
        self.assertEqual(self.source.read_text(), SOURCE)
        reopened = self.maintainer([])
        reopened.load_checkpoint()
        self.assertEqual(reopened.active_project, "todo")

    def test_terminal_freezes_discovered_command_and_verifies_new_tests_separately(self):
        tests = self.workspace / "tests"
        tests.mkdir()
        (tests / "test_existing.py").write_text("def test_existing():\n    assert True\n", encoding="utf-8")
        edit = json.dumps({"action": "edit", "target": ".", "message": "", "require_new_tests": True})
        diagnostic = json.dumps({"action": "run_tests", "arguments": {"file": "tests/test_existing.py"}})
        source_only = verify_patch(candidate("strip"))
        submission = json.dumps({"action": "verify_edits", "arguments": {"edits": [
            {"path": "app.py", "old_text": "return value.lower()", "new_text": "return value.strip().lower()"},
            {"path": "tests/test_cleanup.py", "old_text": "",
             "new_text": "from app import normalize\ndef test_cleanup():\n    assert normalize(' A ') == 'a'\n"},
        ]}})
        agent = CodebaseMaintainer(self.config, ScriptedChatModel([edit, diagnostic, source_only, submission]),
                                  sandbox=self.sandbox, note_tool=FakeNotes(), automatic_tests=True)
        runs = []

        def run(candidate, command, _protected):
            self.assertEqual(command[:3], ("python", "-m", "pytest"))
            focused = command[-1].endswith(".py")
            test_exists = (candidate / "tests/test_cleanup.py").exists()
            fixed = ".strip()" in (candidate / "app.py").read_text()
            runs.append((focused, test_exists, fixed))
            passed = not test_exists or fixed
            return {"status": "passed" if passed else "failed", "exit_code": int(not passed), "output": "test run"}

        with patch.object(self.sandbox, "ensure_ready"), patch.object(self.sandbox, "_run_test", side_effect=run):
            reply = agent.converse("去掉输入首尾的空格，并补一个测试")
        self.assertEqual(runs, [(False, False, False), (True, False, False), (False, False, True),
                                (True, True, False), (True, True, True)])
        self.assertEqual(reply.repair["verification"]["new_test_before:tests/test_cleanup.py"]["status"], "failed")
        self.assertEqual(reply.repair["verification"]["new_test_after:tests/test_cleanup.py"]["status"], "passed")
        self.assertTrue((tests / "test_cleanup.py").exists())
        self.assertIn("pytest", agent.session_status()["test_plan"]["command"])
        self.assertTrue(any((event.metadata.get("submission_error") or {}).get("kind") == "missing_test_addition"
                            for event in reply.trace))

    def test_terminal_missing_tests_stops_without_launching_docker(self):
        edit = json.dumps({"action": "edit", "target": ".", "message": ""})
        agent = CodebaseMaintainer(self.config, ScriptedChatModel([edit]), sandbox=self.sandbox,
                                  note_tool=FakeNotes(), automatic_tests=True)
        with patch.object(self.sandbox, "ensure_ready") as ensure:
            reply = agent.converse("修改现有代码")
        ensure.assert_not_called()
        self.assertEqual(reply.repair["error_kind"], "no_tests")
        self.assertEqual(self.source.read_text(), SOURCE)

    def test_project_resume_uses_saved_draft_and_preserves_delivery_conflict(self):
        route = json.dumps({"action": "create", "target": "todo", "message": ""})
        draft = json.dumps({"files": {"todo.py": "VALUE = 1\n"}, "remove": [], "verify": True})
        agent = self.maintainer([route, draft])
        target = self.workspace / "todo"
        with patch.object(agent.project_sandbox, "verify_project", side_effect=RuntimeError("容器暂不可用")):
            with self.assertRaisesRegex(RuntimeError, "暂不可用"):
                agent.converse("新建一个 todo 项目")
        self.assertEqual(agent.session_status()["next"], ("verify_project",))
        self.assertEqual(agent.session_status()["draft_files"], ["todo.py"])
        target.mkdir()
        (target / "user.txt").write_text("手动新增")
        reopened = self.maintainer([])
        passed = {"status": "passed", "exit_code": 0, "output": "OK"}
        with patch.object(reopened.project_sandbox, "verify_project", return_value=passed):
            reply = reopened.resume()
        self.assertEqual(reply.project["status"], "delivery_conflict")
        self.assertEqual((target / "user.txt").read_text(), "手动新增")
        self.assertFalse((target / "todo.py").exists())

    def test_repair_validates_a_single_diff_from_model_prose(self):
        diff = candidate("strip")
        baseline = {"reproduction": {"status": "failed", "exit_code": 1, "output": "original bug"}}
        prose = f"已定位问题。\n\n```diff\n{diff}```\n\n请验证补丁。"
        agent = self.maintainer([json.dumps({"action": "read_file", "arguments": {"relative_path": "app.py"}}),
                                 json.dumps({"answer": prose})])
        passed = {"status": "passed", "patch": diff, "exit_code": 0, "output": "OK"}
        with (patch.object(self.sandbox, "ensure_ready"), patch.object(self.sandbox, "baseline", return_value=baseline),
              patch.object(self.sandbox, "verify", return_value=passed) as verify):
            reply = agent.repair("handle surrounding whitespace")

        verify.assert_called_once()
        self.assertEqual(verify.call_args.args[1], diff)
        self.assertEqual(reply.repair["status"], "passed")

    def test_recount_returns_a_diff_that_plain_git_accepts(self):
        broken = candidate("strip").replace("@@ -1,2 +1,2 @@", "@@ -1,3 +1,3 @@")
        with patch.object(self.sandbox, "_run_test", return_value={"status": "passed", "exit_code": 0, "output": "OK"}):
            result = self.sandbox.verify(self.workspace, broken)
        self.assertEqual(result["status"], "passed", result["output"])
        self.assertTrue(result["patch_recounted"])
        self.sandbox._git_apply(self.workspace, result["patch"], "--check")
        self.assertEqual(self.source.read_text(encoding="utf-8"), SOURCE)

    def test_structured_edits_preserve_bytes_and_run_through_existing_validation(self):
        source = self.workspace / "中文 name.py"
        original = b"def find():\r\n    return 1"
        source.write_bytes(original)
        edits = {"action": "verify_edits", "arguments": {"edits": [
            {"path": source.name, "old_text": "    return 1", "new_text": "    return 2"},
            {"path": source.name, "old_text": "def find():", "new_text": "def find():"},
        ]}}
        agent = self.maintainer([json.dumps(edits)])
        seen = []

        def run(snapshot, command, tests):
            seen.append((snapshot / source.name).read_bytes())
            before_edit = len(seen) == 1
            return {"status": "failed" if before_edit else "passed", "exit_code": int(before_edit), "output": "test output"}

        with patch.object(self.sandbox, "ensure_ready"), patch.object(self.sandbox, "_run_test", side_effect=run):
            reply = agent.repair("return 2")
        self.assertEqual(reply.repair["status"], "passed", reply.repair["output"])
        self.assertEqual(seen, [original, original.replace(b"return 1", b"return 2")])
        self.assertEqual(source.read_bytes(), original)

    def test_structured_edits_reject_ambiguous_and_protected_targets(self):
        self.source.write_text(SOURCE + SOURCE, encoding="utf-8")
        edit = {"path": "app.py", "old_text": "return value.lower()", "new_text": "return value.strip().lower()"}
        with self.assertRaisesRegex(ValueError, "匹配多处"):
            self.sandbox.patch_from_edits(self.workspace, [edit])
        with self.assertRaisesRegex(ValueError, "受保护"):
            self.sandbox.patch_from_edits(self.workspace, [{**edit, "path": "tests/test_app.py"}])

    def test_duplicate_submission_preserves_test_failure_and_does_not_rerun_test(self):
        diff = candidate("lstrip")
        agent = self.maintainer([verify_patch(diff)] * 5)
        failed = {"status": "failed", "patch": diff, "exit_code": 1, "output": "assertion failed"}
        baseline = {"reproduction": {"status": "failed", "exit_code": 1, "output": "baseline"}}
        with (patch.object(self.sandbox, "ensure_ready"), patch.object(self.sandbox, "baseline", return_value=baseline),
              patch.object(self.sandbox, "verify", return_value=failed) as verify):
            reply = agent.repair("trim whitespace")
        verify.assert_called_once()
        self.assertEqual(reply.repair["status"], "failed")
        self.assertEqual(reply.repair["submission_error"]["kind"], "duplicate")
        self.assertIn("assertion failed", agent.model.calls[-1][-1].content)
        self.assertIn('"kind": "duplicate"', agent.model.calls[-1][-1].content)

    def test_exhausted_reads_still_allow_a_corrected_submission(self):
        read = lambda **arguments: json.dumps({"action": "read_file", "arguments": arguments})
        noop = json.dumps({"action": "verify_edits", "arguments": {"edits": [
            {"path": "app.py", "old_text": "return value.lower()", "new_text": "return value.lower()"},
        ]}})
        agent = self.maintainer([
            read(file="app.py"), read(path="app.py"), read(relative_path="app.py", end_line=1),
            read(relative_path="app.py", start_line=2), read(relative_path="app.py", end_line=2),
            noop, read(relative_path="app.py", start_line=3), verify_patch(candidate("strip")),
        ])
        baseline = {"reproduction": {"status": "failed", "exit_code": 1,
                                      "output": '  File "/work/app.py", line 2\nAssertionError: original failure'}}
        passed = {"status": "passed", "patch": candidate("strip"), "exit_code": 0, "output": "OK"}
        with (patch.object(self.sandbox, "ensure_ready"), patch.object(self.sandbox, "baseline", return_value=baseline),
              patch.object(self.sandbox, "verify", return_value=passed) as verify):
            reply = agent.repair("trim whitespace")

        verify.assert_called_once()
        self.assertEqual(reply.repair["status"], "passed")
        prompts = [call[-1].content for call in agent.model.calls]
        self.assertIn("app.py:1-2（堆栈行 2）", prompts[0])
        self.assertIn("剩余取证 4 次", prompts[1])
        self.assertIn('"relative_path": "app.py"', prompts[2])
        self.assertIn("剩余取证 0 次，补丁提交 5 次，实际测试 3 次", prompts[5])
        self.assertIn("AssertionError: original failure", prompts[6])
        self.assertIn("源码无变化", prompts[6])
        self.assertIn("本轮请直接提交实质性的补丁", prompts[7])

    def test_failed_tests_drive_retry_on_same_snapshot_then_return_verified_diff(self):
        if shutil.which("git") is None:
            self.skipTest("requires git apply")
        (self.workspace / ".env").write_text("SECRET=fixture", encoding="utf-8")
        (self.workspace / "link").symlink_to(self.source)
        self.source.chmod(0o600)
        replies = [verify_patch(candidate("lstrip")), verify_patch(candidate("strip")), "next turn"]
        agent = self.maintainer(replies)
        executed = []

        def run(command, timeout):
            self.assertEqual(timeout, 90)
            self.assertIn("--network=none", command)
            self.assertIn("--read-only", command)
            self.assertEqual(command[-3:], ["python", "-m", "unittest"])
            mount = command[command.index("--mount") + 1]
            self.assertTrue(mount.endswith(",target=/input,readonly"))
            staged = Path(mount.removeprefix("type=bind,source=").split(",target=", 1)[0])
            self.assertFalse((staged / ".env").exists())
            self.assertFalse((staged / "link").exists())
            self.assertFalse((staged / "state").exists())
            self.assertEqual((staged / "app.py").stat().st_mode & 0o777, 0o644)
            executed.append((staged, (staged / "app.py").read_text(encoding="utf-8")))
            return (1, "AssertionError: trailing whitespace") if len(executed) <= 2 else (0, "OK")

        with patch.object(self.sandbox, "ensure_ready"), patch.object(self.sandbox, "_run", side_effect=run):
            reply = agent.repair("handle surrounding whitespace")
        self.assertEqual(reply.repair["status"], "passed")
        self.assertEqual(reply.repair["patch"], candidate("strip"))
        self.assertEqual(reply.repair["changed_files"], [{"path": "app.py", "added": 1, "removed": 1}])
        self.assertEqual(reply.baseline["reproduction"]["status"], "failed")
        self.assertIn("AssertionError: trailing whitespace", agent.model.calls[1][-1].content)
        self.assertEqual([item[1] for item in executed],
                         [SOURCE, *(SOURCE.replace(".lower", f".{method}().lower") for method in ("lstrip", "strip"))])
        self.assertTrue(all(not location.exists() for location, _ in executed))
        self.assertEqual(self.source.read_text(encoding="utf-8"), SOURCE)
        self.assertEqual(self.source.stat().st_mode & 0o777, 0o600)
        self.assertEqual(list((self.config.state_dir / "repair-snapshots").iterdir()), [])
        self.assertIsNone(agent.analyze_issue("follow-up").repair)

    def test_resume_keeps_snapshot_and_does_not_rerun_completed_test(self):
        failed = {"status": "failed", "patch": candidate("lstrip"), "exit_code": 1, "output": "wrong"}
        passed = {"status": "passed", "patch": candidate("strip"), "exit_code": 0, "output": "OK"}
        agent = self.maintainer([verify_patch(failed["patch"])])
        baseline = {"reproduction": {"status": "failed", "exit_code": 1, "output": "original bug"}}
        with (patch.object(self.sandbox, "ensure_ready"), patch.object(self.sandbox, "baseline", return_value=baseline) as before,
              patch.object(self.sandbox, "verify", side_effect=[failed, passed]) as verify):
            with self.assertRaises(ModelNotConfigured):
                agent.repair("whitespace")
            self.source.write_text("host file changed", encoding="utf-8")
            resumed = self.maintainer([verify_patch(passed["patch"])])
            self.assertEqual(resumed.resume().repair["patch"], passed["patch"])
        before.assert_called_once()
        self.assertEqual(verify.call_count, 2)
        self.assertEqual(verify.call_args_list[0].args[0], verify.call_args_list[1].args[0])
        self.assertIn('"exit_code": 1', resumed.model.calls[0][-1].content)

    def test_timeout_cleans_up_the_container(self):
        if shutil.which("git") is None:
            self.skipTest("requires git apply")
        with patch.object(self.sandbox, "_run", return_value=(None, "timed out")) as run, \
                patch.object(self.sandbox, "_remove") as remove:
            result = self.sandbox.verify(self.workspace, candidate("strip"))
        self.assertEqual(result["status"], "timeout")
        command = run.call_args.args[0]
        remove.assert_called_once_with(command[command.index("--name") + 1])

    def test_unittest_without_discovered_tests_is_not_a_pass(self):
        if shutil.which("git") is None:
            self.skipTest("requires git apply")
        with patch.object(self.sandbox, "_run", return_value=(0, "Ran 0 tests in 0.001s\n\nOK")):
            result = self.sandbox.verify(self.workspace, candidate("strip"))
        self.assertEqual(result["status"], "sandbox_error")

    def test_invalid_patch_never_starts_a_container(self):
        if shutil.which("git") is None:
            self.skipTest("requires git apply")
        unsafe = candidate("strip").replace("a/app.py", "a/../app.py").replace("b/app.py", "b/../app.py")
        with patch.object(self.sandbox, "_run") as run:
            self.assertEqual(self.sandbox.verify(self.workspace, unsafe)["status"], "patch_rejected")
            run.assert_not_called()

    def test_passing_baseline_stops_before_model(self):
        agent = self.maintainer([])
        with patch.object(self.sandbox, "ensure_ready"), patch.object(self.sandbox, "_run", return_value=(0, "OK")):
            reply = agent.repair("already fixed")
        self.assertEqual(reply.repair["status"], "not_reproduced")
        self.assertEqual(agent.model.calls, [])

    def test_unrelated_baseline_failure_stops_before_model(self):
        self.sandbox.expected_failure = "AssertionError: wrong result"
        agent = self.maintainer([])
        with patch.object(self.sandbox, "ensure_ready"), patch.object(self.sandbox, "_run", return_value=(1, "ImportError")):
            reply = agent.repair("broken calculation")
        self.assertEqual(reply.repair["status"], "not_reproduced")
        self.assertEqual(agent.model.calls, [])

    def test_protected_tests_are_frozen_and_shared_by_all_runs(self):
        external = self.tmp / "judge"
        external.mkdir()
        script = external / "test_app.py"
        script.write_text("initial judge")
        self.sandbox.protected_tests = external
        self.sandbox.regression_command = ("python", "regression.py")
        agent = self.maintainer([verify_patch(candidate("strip"))])
        mounted_paths = []

        def run(command, timeout):
            mount = next(arg for arg in command if arg.endswith(",target=/agent-tests,readonly"))
            mounted = Path(mount.removeprefix("type=bind,source=").split(",target=", 1)[0])
            mounted_paths.append(mounted)
            self.assertEqual((mounted / "test_app.py").read_text(), "initial judge")
            script.write_text("edited while repairing")
            return (1, "original bug") if len(mounted_paths) == 1 else (0, "OK")

        with patch.object(self.sandbox, "ensure_ready"), patch.object(self.sandbox, "_run", side_effect=run):
            reply = agent.repair("whitespace")
        self.assertEqual(reply.repair["status"], "passed")
        self.assertEqual(len(mounted_paths), 4)
        self.assertEqual(len(set(mounted_paths)), 1)
        self.assertFalse(mounted_paths[0].exists())

    def test_regression_failure_prevents_acceptance(self):
        self.sandbox.regression_command = ("python", "regressions.py")
        with patch.object(self.sandbox, "_run", side_effect=[(0, "OK"), (1, "regression broke")]):
            result = self.sandbox.verify(self.workspace, candidate("strip"))
        self.assertEqual(result["status"], "regression_failed")
        self.assertEqual(result["verification"]["reproduction"]["status"], "passed")

    def test_repair_rejects_edits_to_tests_and_outside_allowed_paths(self):
        self.sandbox.allowed_paths = ("app.py",)
        for path in ("tests/test_app.py", "other.py"):
            with self.subTest(path=path), patch.object(self.sandbox, "_run") as run:
                result = self.sandbox.verify(self.workspace, candidate("strip").replace("app.py", path))
                self.assertEqual(result["status"], "patch_rejected")
                run.assert_not_called()

    def test_symbol_tool_reports_definition_reference_and_test(self):
        self.source.write_text(SOURCE + "\ndef local_call():\n    return normalize('A')\n")
        (self.workspace / "main.py").write_text("from app import normalize as clean\n\ndef run():\n    return clean('A')\n")
        (self.workspace / "test_app.py").write_text("from app import normalize\n\ndef test_query():\n    assert normalize('A') == 'a'\n")
        model = ScriptedChatModel(['{"action":"find_symbol","arguments":{"symbol":"app.normalize","limit":2}}', 'done'])
        agent = CodebaseMaintainer(self.config, model, note_tool=FakeNotes())
        agent.analyze_issue("locate app.normalize")
        prompt = model.calls[-1][-1].content
        for path in ("app.py", "main.py", "test_app.py"):
            self.assertIn(path, prompt)

        refs = PythonSymbols(self.workspace, self.config.state_dir).find("app.normalize")["references"]
        self.assertIn("app.local_call", [item["symbol"] for item in refs])

    def test_delivery_stages_patch_in_a_separate_worktree(self):
        def git(*args):
            return subprocess.run(["git", "-C", str(self.workspace), *args], check=True, text=True, capture_output=True).stdout.strip()

        git("init", "-q")
        git("add", "app.py")
        git("-c", "user.email=test@example.invalid", "-c", "user.name=Test", "commit", "-qm", "base")
        result = {"status": "passed", "patch": candidate("strip"), "base_commit": base_commit(self.workspace),
                  "base_digest": tree_digest(self.workspace / "__missing__")}
        with self.assertRaisesRegex(ValueError, "测试的快照不同"):
            deliver(self.workspace, result, "repair/test", self.tmp / "delivered")
        snapshot = self.tmp / "source"
        snapshot.mkdir()
        shutil.copy2(self.source, snapshot / "app.py")
        result["base_digest"] = tree_digest(snapshot)
        target = deliver(self.workspace, result, "repair/test", self.tmp / "delivered")
        self.assertIn(".strip()", (target / "app.py").read_text())
        self.assertEqual(git("status", "--porcelain"), "")
        self.assertEqual(self.source.read_text(), SOURCE)

    def test_repoqa_adapter_scores_annotated_path_and_utf8_span(self):
        source = '# 中文\n\ndef normalize_query(value):\n    return value.strip().lower()\n'
        start = len(source[:source.index("def ")].encode())
        repository = {"repo": "example/search", "content": {"src/a.py": source}, "needles": [
            {"description": "normalize query strip lower", "name": "normalize_query", "path": "src/a.py",
             "start_byte": start, "end_byte": len(source.rstrip().encode())},
            {"description": "unrelated lexical word", "name": "normalize_query", "path": "src/a.py",
             "start_byte": start, "end_byte": len(source.rstrip().encode())},
        ]}
        dataset = self.tmp / "dataset.json.gz"
        with gzip.open(dataset, "wt", encoding="utf-8") as stream:
            json.dump({"python": [repository]}, stream)
        metrics = evaluate_repoqa(dataset, 1)
        self.assertEqual((metrics["cases"], metrics["hit_at_k"], metrics["mrr_at_k"]), (2, 0.5, 0.5))

    def test_repoqa_reranks_only_the_selected_pool_and_preserves_its_coverage(self):
        from types import SimpleNamespace

        from context_lab.types import SearchHit

        target = "def tokenize(text):\n    return text.split()"
        distractor = 'def cleanup(text):\n    """tokenize tokenize tokenize"""\n    return text.strip()'
        repository = {"repo": "example/code", "content": {"a.py": target, "b.py": distractor}, "needles": [
            {"description": "tokenize", "name": "tokenize", "path": "a.py", "start_byte": 0, "end_byte": len(target)},
        ]}
        dataset = self.tmp / "dataset.json"
        dataset.write_text(json.dumps({"python": [repository]}), encoding="utf-8")

        class Vectors:
            client = SimpleNamespace(config=SimpleNamespace(model="test-vector"), calls=0)

            def prepare(self, chunks):
                return object()

            def search(self, query, chunks, matrix, limit):
                self.client.calls += 1
                return [SearchHit(chunk, 1.0) for chunk in chunks if chunk.source == "a.py"]

        class Reranker:
            config = SimpleNamespace(model="test-reranker")

            def __init__(self, expected_sources):
                self.expected_sources = expected_sources

            def rerank(self, query, hits, limit):
                if {hit.chunk.source for hit in hits} != self.expected_sources:
                    raise AssertionError("reranker received candidates from the wrong source")
                return list(reversed(hits))[:limit]

        for source, expected_sources in (("hybrid", {"a.py", "b.py"}), ("vector", {"a.py"})):
            with self.subTest(source=source):
                metrics = evaluate_repoqa(dataset, k=1, vectors=Vectors(),
                                          reranker=Reranker(expected_sources), candidates=2, rerank_source=source)
                self.assertEqual(metrics["rerank_source"], source)
                self.assertEqual(set(metrics["runs"]), {"bm25", "vector", "hybrid", f"{source}_rerank"})
                reranked = metrics["runs"][f"{source}_rerank"]
                self.assertEqual(reranked["hit_at_k"], 1.0 if source == "vector" else 0.0)
                self.assertEqual(reranked["candidate_hit_rate"], 1.0)
        self.assertEqual(Vectors.client.calls, 2)

        compared = evaluate_repoqa(
            dataset, k=1, vectors=Vectors(), candidates=2, comparison_source="hybrid",
            comparisons=(RerankComparison("compared_top2", 2, Reranker({"a.py", "b.py"})),),
            rerank_workers=2,
        )["runs"]
        self.assertEqual(compared["hybrid_top2"]["hit_at_k"], 1.0)
        self.assertEqual(compared["compared_top2"]["hit_at_k"], 0.0)
        self.assertEqual(compared["compared_top2"]["candidate_hit_rate"], compared["hybrid_top2"]["candidate_hit_rate"])

    def test_repoqa_query_augmentation_reuses_generated_variants_and_limits_rerank_pool(self):
        from types import SimpleNamespace

        from context_lab.types import SearchHit

        target = "def tokenize(text):\n    return text.split()"
        dataset = self.tmp / "dataset.json"
        dataset.write_text(json.dumps({"python": [{
            "repo": "example/code", "content": {"a.py": target, "b.py": "def cleanup(text):\n    return text.strip()"},
            "needles": [{"description": "tokenize", "name": "tokenize", "path": "a.py", "start_byte": 0, "end_byte": len(target)}],
        }]}), encoding="utf-8")

        class Vectors:
            client = SimpleNamespace(config=SimpleNamespace(model="fixture"))
            inputs = []

            def prepare(self, chunks):
                return object()

            def score_queries(self, queries, matrix):
                self.inputs.extend(queries)
                return [0.1 if name == "tokenize" else 0.9 for name in queries]

            def rank(self, chunks, score, limit):
                hits = [SearchHit(chunks[0], score), SearchHit(chunks[1], 0.2)]
                return sorted(hits, key=lambda hit: -hit.score)[:limit]

        class Reranker:
            config = SimpleNamespace(model="fixture")

            def rerank(self, query, hits, limit):
                if query != "tokenize" or len(hits) != 1:
                    raise AssertionError("reranking must use the original query and configured candidate limit")
                return hits

        model = ScriptedChatModel(['{"queries":["split text","split on whitespace"],"hypothetical_document":"Splits incoming text into words."}'])
        vectors = Vectors()
        metrics = evaluate_repoqa(dataset, k=1, candidates=2, vectors=vectors, query_model=model,
                                  hyde=True, multi_query=True, reranker=Reranker(),
                                  rerank_source="enhanced", rerank_candidates=1, query_cache=self.tmp / "queries.json")
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(vectors.inputs, ["tokenize", "Splits incoming text into words.", "split text", "split on whitespace"])
        self.assertEqual(metrics["runs"]["vector"]["hit_at_k"], 0)
        for name in ("vector_hyde", "vector_multi_query", "vector_hyde_multi_query", "vector_hyde_multi_query_rerank"):
            self.assertEqual(metrics["runs"][name]["hit_at_k"], 1)
        cached = evaluate_repoqa(dataset, k=1, candidates=2, vectors=vectors, query_model=ScriptedChatModel([]),
                                 hyde=True, multi_query=True, query_cache=self.tmp / "queries.json")
        self.assertEqual(cached["runs"]["vector_hyde_multi_query"]["hit_at_k"], 1)

    def test_model_adapter_sends_configured_non_streaming_request(self):
        response = io.BytesIO(b'{"choices":[{"message":{"content":"ready"}}]}')
        model = HTTPChatModel("test", "https://example.invalid/v1", api_key="dummy")
        with patch("context_lab.models.urlopen", return_value=response) as send:
            self.assertEqual(model.generate([Message("user", "question")]), "ready")
        request = send.call_args.args[0]
        self.assertEqual(request.full_url, "https://example.invalid/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer dummy")
        self.assertEqual(json.loads(request.data), {"model": "test", "stream": False,
                                                   "messages": [{"role": "user", "content": "question"}]})

    def test_ecnu_max_profile_enforces_the_application_output_budget_in_the_request(self):
        config = AppConfig.for_model("ecnu-max")
        self.assertEqual((config.context.input_budget, config.max_output_tokens), (32768, 8192))
        response = io.BytesIO(b'{"choices":[{"message":{"content":"ready"}}]}')
        model = HTTPChatModel("ecnu-max", "https://example.invalid/v1", api_key="dummy",
                              max_output_tokens=config.max_output_tokens)
        with patch("context_lab.models.urlopen", return_value=response) as send:
            model.generate([Message("user", "question")])
        self.assertEqual(json.loads(send.call_args.args[0].data)["max_tokens"], 8192)
