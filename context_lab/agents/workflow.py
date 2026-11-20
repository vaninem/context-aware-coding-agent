"""LangGraph 节点编排只读取证与隔离补丁验证。"""

import json
import re
from contextlib import closing
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from subprocess import TimeoutExpired
from time import perf_counter
from typing import Any, Literal, TYPE_CHECKING, TypedDict

from langgraph.graph import END, START, StateGraph

from ..errors import ExercisePending
from ..rag.types import DEFAULT_EVIDENCE_TOKENS, RAGSearchOptions
from ..tokens import ApproxTokenCounter
from ..tools.project_files import (
    apply_verified_patch, normalize_project_root, project_path, publish_project, update_files, write_files,
)
from ..tools.sandbox import MAX_PATCH_CHARS, test_runner, tree_digest
from ..tools.rag import RAGTool
from ..tools.symbols import PythonSymbols
from ..tools.terminal import TerminalTool
from ..tools.test_discovery import TestPlan, discover_tests
from ..types import ContextBuildResult, ContextCompactionState, ContextPacket, Message, ProjectResult, RepairResult, TestRun, TraceEvent
from .interactive import BUILD_PROMPT, generate_with_retry, read_project_update, route_request

if TYPE_CHECKING:
    from .base import ContextAwareAgent


WorkflowMode = Literal["chat", "explore", "issue", "plan", "repair", "route", "clarify", "create"]
MAX_TOOL_CALLS = 4
MAX_REPAIR_ATTEMPTS = 3
MAX_REPAIR_SUBMISSIONS = 5
MAX_PROJECT_STEPS = 8
VERIFY_PATCH = "verify_patch"
VERIFY_EDITS = "verify_edits"
RUN_TESTS = "run_tests"
INVALID_RESPONSE = "invalid_response"
REPAIR_SUBMISSIONS = (VERIFY_PATCH, VERIFY_EDITS, INVALID_RESPONSE)
TRACE_FRAME = re.compile(r'^[ \t]*File "([^"\n]+)", line ([1-9][0-9]{0,8})\b', re.MULTILINE)
DIFF_BLOCK = re.compile(r"^```(?:diff|patch)[ \t]*\r?\n(.*?)^```[ \t]*(?:\r?$)", re.MULTILINE | re.DOTALL)
TOOL_ARGUMENTS = {
    "list_files": ({"relative_path", "pattern"}, set()),
    "read_file": ({"relative_path", "start_line", "end_line"}, {"relative_path"}),
    "search": ({"query", "relative_path", "limit"}, {"query"}),
    "rag_search": ({"query", "limit"}, {"query"}),
    "find_symbol": ({"symbol", "limit"}, {"symbol"}),
    RUN_TESTS: ({"file"}, set()),
}


class ToolArgumentsError(ValueError):
    """模型可以订正的只读工具参数错误。"""


class TurnState(TypedDict, total=False):
    turn_id: str
    requested_rag_mode: str
    index_pattern: str
    embedding_profile: str | None
    rag_mode: str
    query: str
    task_id: str | None
    mode: WorkflowMode
    instructions: str
    context: ContextBuildResult | None
    compaction: ContextCompactionState
    observations: list[ContextPacket]
    calls: list[dict[str, Any]]
    action: dict[str, Any] | None
    answer: str
    events: list[TraceEvent]
    sandbox_settings: dict | None
    test_plan: TestPlan | None
    require_new_tests: bool
    repair: RepairResult | None
    tested_patches: list[str]
    validation_runs: int
    submission_error: dict[str, str] | None
    read_correction_used: bool
    submission_only: bool
    baseline_evidence: str
    baseline: dict[str, TestRun] | None
    base_commit: str | None
    prior_context: ContextPacket | None
    history_start: int
    user_preference: str
    stream_answer: bool
    context_budget: dict
    interactive_turn: bool
    target: str
    project_settings: dict | None
    project_files: dict[str, str]
    project_steps: int
    project_verify: bool
    project_run: TestRun | None
    project_error: str
    project: ProjectResult | None


class TurnWorkflow:
    def __init__(self, agent: "ContextAwareAgent") -> None:
        self.agent = agent
        self.symbols = None
        self.progress = getattr(agent, "progress", None)

    def _emit(self, stage: str, message: str, **metadata: Any) -> TraceEvent:
        event = TraceEvent(stage, message, metadata)
        if self.progress is not None:
            self.progress(event)
        return event

    def compile(self, checkpointer=None):
        graph = StateGraph(TurnState)
        graph.add_node("prepare_context", self.prepare_context)
        graph.add_node("check_baseline", self.check_baseline)
        graph.add_node("decide", self.decide)
        graph.add_node("render_answer", self.render_answer)
        graph.add_node("execute_tool", self.execute_tool)
        graph.add_node("route_request", self.route)
        graph.add_node("freeze_repair", self.freeze_repair)
        graph.add_node("discover_tests", self.discover_repair_tests)
        graph.add_node("draft_project", self.draft_project)
        graph.add_node("verify_project", self.verify_project)
        graph.add_node("finish_project", self.finish_project)
        graph.add_conditional_edges(START, lambda state: "route_request" if state["mode"] == "route" else "prepare_context",
                                    ["route_request", "prepare_context"])
        graph.add_edge("route_request", "prepare_context")
        graph.add_conditional_edges("prepare_context", self.after_context,
                                    ["execute_tool", "decide", "check_baseline", "freeze_repair", "draft_project", END])
        graph.add_edge("freeze_repair", "discover_tests")
        graph.add_edge("discover_tests", "check_baseline")
        graph.add_conditional_edges("draft_project", self.after_draft, ["draft_project", "verify_project", "finish_project"])
        graph.add_conditional_edges("verify_project", self.after_project_test, ["draft_project", "finish_project"])
        graph.add_conditional_edges("finish_project", self.after_delivery, ["render_answer", END])
        graph.add_conditional_edges("check_baseline", lambda state: "finish_repair" if state.get("repair") else "decide",
                                    ["finish_repair", "decide"])
        graph.add_conditional_edges("decide", self.after_decide, ["execute_tool", "decide", "finish_repair", "render_answer", END])
        graph.add_conditional_edges("execute_tool", self.after_tool,
                                    ["decide", "finish_repair"])
        graph.add_node("finish_repair", self.finish_repair)
        graph.add_conditional_edges("finish_repair", self.after_delivery, ["render_answer", END])
        graph.add_edge("render_answer", END)
        return graph.compile(checkpointer=checkpointer)

    @staticmethod
    def after_context(state: TurnState) -> str:
        if state["mode"] == "create":
            return "draft_project"
        if state["mode"] == "clarify":
            return END
        if state["mode"] == "repair":
            return "freeze_repair" if state.get("interactive_turn") else "check_baseline"
        return "execute_tool" if state["action"] else "decide"

    def route(self, state: TurnState) -> TurnState:
        self._emit("routing_started", "正在判断本轮任务")
        action, target, answer, require_new_tests, preference = route_request(self.agent, state["query"])
        mode = {"analyze": "issue", "edit": "repair", "create": "create", "clarify": "clarify"}[action]
        event = self._emit("model", "已判断本轮任务", mode=action, location=target,
                           require_new_tests=require_new_tests)
        sandbox = self.agent.project_sandbox if mode == "create" else None
        update: TurnState = {"mode": mode, "target": target, "answer": answer, "events": [*state["events"], event],
                             "require_new_tests": require_new_tests, "user_preference": preference,
                             "stream_answer": (self.agent.answer_output is not None
                                               and callable(getattr(self.agent.model, "generate_stream", None))),
                             "project_settings": sandbox.settings() if sandbox is not None else None}
        if self.agent.terminal_memory is not None:
            update.update(self.agent.scope_terminal_turn(target, state.get("prior_context")))
        return update

    def _workspace(self, state: TurnState) -> Path:
        return project_path(self.agent.config.workspace, state.get("target", "."))

    def _project_relative(self, path: str, state: TurnState) -> str:
        """兼容模型从工作区根部拼出的路径，同时保留项目内同名目录。"""
        target = state.get("target", ".")
        if target == "." or (path != target and not path.startswith(target + "/")):
            return path
        root = (self.agent._snapshot_path(state["turn_id"]) if state["mode"] == "repair"
                else self._workspace(state))
        return path if (root / target).exists() else path[len(target):].removeprefix("/") or "."

    def freeze_repair(self, state: TurnState) -> TurnState:
        return self.agent._freeze_repair(state["turn_id"], self._workspace(state),
                                         defer_sandbox=self.agent.automatic_tests)

    def discover_repair_tests(self, state: TurnState) -> TurnState:
        sandbox = self.agent.sandbox
        if self.agent.automatic_tests:
            plan = discover_tests(self.agent._snapshot_path(state["turn_id"]))
        else:
            plan = {"command": list(sandbox.test_command), "source": "显式配置"}
        event = self._emit("test_plan", "已确定测试计划" if plan["command"] else "未发现 Python 测试入口", **plan)
        return {"test_plan": plan, "events": [*state["events"], event]}

    def _repair_test_command(self, state: TurnState) -> tuple[str, ...]:
        plan = state.get("test_plan")
        return tuple(plan["command"]) if plan is not None else self.agent.sandbox.test_command

    def draft_project(self, state: TurnState) -> TurnState:
        """把一批文件更新持久化到图状态，让测试失败时可以接着修改。"""
        sandbox = self.agent.project_sandbox
        if sandbox is None:
            raise RuntimeError("生成项目需要配置 Docker 测试环境")
        step = state["project_steps"] + 1
        self._emit("project_draft_started", "正在生成项目文件", location=state["target"], submission=step)
        counter = getattr(self.agent.context_builder, "token_counter", None) or ApproxTokenCounter()
        budget = self.agent.config.context.input_budget
        files = state["project_files"]
        per_file = min(1400, max(64, budget // (2 * max(1, len(files)))))
        feedback = {
            "task": counter.truncate(state["query"], max(1, budget // 3)),
            "test_command": list(sandbox.test_command),
            "steps_left": MAX_PROJECT_STEPS - step, "tests_left": MAX_REPAIR_ATTEMPTS - state["validation_runs"],
            "files": {name: counter.truncate(text, per_file) for name, text in files.items()},
            "previous_error": state["project_error"],
            "test_output": (state["project_run"] or {}).get("output", "")[-3000:],
        }
        notes = "\n\n".join(packet.content for packet in state["context"].selected
                            if packet.metadata.get("type") != "system_instruction")
        feedback["context"] = counter.truncate(notes, max(1, budget // 5))
        reply = generate_with_retry(self.agent.model, [Message("system", BUILD_PROMPT),
                                                       Message("user", json.dumps(feedback, ensure_ascii=False))])
        events = [*state["events"], TraceEvent("model", "已获得项目文件更新")]
        try:
            written, removed, verify = read_project_update(reply)
            updated = update_files(files, written, removed, self._workspace(state))
            if verify and not updated:
                raise ValueError("提交测试前需要项目文件")
            if verify:
                updated = normalize_project_root(updated, state["target"])
        except (AttributeError, ValueError, UnicodeError) as exc:
            self._emit("project_draft_rejected", "文件格式无效，正在重新生成", status=type(exc).__name__)
            return {"project_steps": step, "project_verify": False,
                    "project_error": f"文件提交被拒绝：{exc}", "events": events}
        event = self._emit("project_draft", "项目草稿已保存", location=state["target"], files=len(updated))
        return {"project_files": updated, "project_steps": step, "project_verify": verify,
                "project_error": "", "events": [*events, event]}

    @staticmethod
    def after_draft(state: TurnState) -> str:
        if state["project_verify"]:
            return "verify_project"
        return "finish_project" if state["project_steps"] >= MAX_PROJECT_STEPS else "draft_project"

    def verify_project(self, state: TurnState) -> TurnState:
        attempt = state["validation_runs"] + 1
        self._emit("verification_started", "正在运行新项目测试", submission=attempt)
        with self.agent.project_sandbox.scratch("agent-project-test-") as folder:
            candidate = (Path(folder) / "project").resolve()
            candidate.mkdir()
            write_files(candidate, state["project_files"])
            run = self.agent.project_sandbox.verify_project(candidate)
        event = self._emit("tool", "新项目测试完成", tool="verify_project", status=run["status"],
                           submission=attempt, ran_tests=run["status"] != "sandbox_error")
        return {"project_run": run, "validation_runs": attempt, "events": [*state["events"], event]}

    @staticmethod
    def after_project_test(state: TurnState) -> str:
        if (state["project_run"]["status"] in {"passed", "sandbox_error"}
                or state["validation_runs"] >= MAX_REPAIR_ATTEMPTS or state["project_steps"] >= MAX_PROJECT_STEPS):
            return "finish_project"
        return "draft_project"

    def finish_project(self, state: TurnState) -> TurnState:
        files, run = state["project_files"], state["project_run"]
        target = self.agent.config.workspace / state["target"]
        result: ProjectResult = {"status": "failed", "target": str(target),
                                 "files": sorted(files), "verification": run}
        if run is None or run["status"] != "passed" or not state["project_verify"]:
            detail = state["project_error"] or (run or {}).get("output", "")[-1200:]
            return {"project": result, "answer": "项目未通过验证。\n" + detail}
        try:
            target = publish_project(self.agent.config.workspace, state["target"], files)
        except (OSError, ValueError) as exc:
            return {"project": {**result, "status": "delivery_conflict"}, "answer": f"项目已通过测试，交付未完成。{exc}"}
        event = self._emit("project_delivered", "项目已通过测试并交付", location=str(target))
        return {"project": {**result, "status": "generated"}, "answer": f"项目已生成并通过测试，共 {len(files)} 个文件。",
                "events": [*state["events"], event]}

    def check_baseline(self, state: TurnState) -> TurnState:
        snapshot = self.agent._snapshot_path(state["turn_id"])
        command = self._repair_test_command(state)
        if not command:
            run: TestRun = {"status": "sandbox_error", "exit_code": None,
                            "output": "未发现 Python 测试入口。如果项目有自定义测试入口，可通过 --test-command 指定。",
                            "error_kind": "no_tests"}
            return {"baseline": {"reproduction": run}, "repair": {**run, "patch": "", "phase": "baseline_reproduction"}}
        self._emit("baseline_started", "正在运行修改前的测试")
        try:
            sandbox = self.agent.sandbox
            if state.get("interactive_turn") and self.agent.automatic_tests:
                sandbox.ensure_ready()
            kwargs = {"command": command} if command != sandbox.test_command else {}
            if state.get("interactive_turn") and not sandbox.expected_failure:
                kwargs["check_passing_regression"] = True
            baseline = sandbox.baseline(snapshot, **kwargs)
        except (OSError, RuntimeError) as exc:
            baseline = {"reproduction": {"status": "sandbox_error", "exit_code": None,
                                         "output": f"准备复现环境失败：{exc}"}}
        events = [*state["events"], self._emit("baseline", "修改前测试完成", **{
            name: result["status"] for name, result in baseline.items()
        })]
        reproduction = baseline["reproduction"]
        expected = self.agent.sandbox.expected_failure
        if reproduction["status"] == "failed" and expected and expected not in reproduction["output"]:
            output = f"基线失败日志未出现指定标记：{expected}\n{reproduction['output']}"
            return {"baseline": baseline, "events": events, "repair": {
                **reproduction, "status": "not_reproduced", "patch": "", "phase": "baseline_reproduction", "output": output,
            }}
        can_edit_passing = state.get("interactive_turn") and not expected and reproduction["status"] == "passed"
        if (reproduction["status"] == "failed" or can_edit_passing) and (
            "regression" not in baseline or baseline["regression"]["status"] == "passed"
        ):
            evidence = self._read_traceback_source(snapshot, reproduction["output"]) if reproduction["status"] == "failed" else ""
            if evidence:
                events.append(self._emit("baseline_source", "已读取复现堆栈指向的仓库源码",
                                         location=evidence.split("\n", 1)[0]))
            requested = state.get("requested_rag_mode", "lexical")
            mode = "lexical"
            if requested != "lexical":
                self._emit("rag_index_started", "正在准备源码向量索引", mode=requested)
                try:
                    symbols = self._repair_symbols(snapshot, requested, self._workspace(state))
                    status = symbols.rag.prepare_index()
                except (OSError, RuntimeError) as exc:
                    events.append(self._emit("rag_index", "向量索引准备失败，使用词法检索",
                                             mode=mode, reason=type(exc).__name__))
                else:
                    self.symbols = symbols
                    if not (status.pending or status.running or status.failed):
                        mode = requested
                    events.append(self._emit("rag_index", "仓库检索准备完成", mode=mode, ready=status.ready,
                                             pending=status.pending, running=status.running, failed=status.failed))
            return {"baseline": baseline, "events": events, "baseline_evidence": evidence, "rag_mode": mode}
        phase = "regression" if "regression" in baseline and baseline["regression"]["status"] != "passed" else "reproduction"
        run = baseline[phase]
        status = run["status"]
        if status == "passed":
            status = "not_reproduced"
        elif phase == "regression" and status == "failed":
            status = "regression_failed"
        return {"baseline": baseline, "events": events, "repair": {**run, "status": status, "patch": "", "phase": f"baseline_{phase}"}}

    @staticmethod
    def after_decide(state: TurnState) -> str:
        if state["action"]:
            return "execute_tool"
        if state.get("submission_only"):
            return "decide"
        if state["mode"] == "repair":
            return "finish_repair"
        return "render_answer" if state.get("stream_answer") else END

    @staticmethod
    def after_delivery(state: TurnState) -> str:
        delivered = (state.get("repair") or {}).get("applied_to") or (state.get("project") or {}).get("status") == "generated"
        return "render_answer" if delivered and state.get("stream_answer") else END

    @staticmethod
    def after_tool(state: TurnState) -> Literal["decide", "finish_repair"]:
        if state["mode"] == "repair" and state["calls"][-1]["action"] in REPAIR_SUBMISSIONS:
            result = state.get("repair")
            if (result and result["status"] in {"passed", "sandbox_error"}
                    or sum(call["action"] in REPAIR_SUBMISSIONS for call in state["calls"]) >= MAX_REPAIR_SUBMISSIONS
                    or state.get("validation_runs", 0) >= MAX_REPAIR_ATTEMPTS):
                return "finish_repair"
        return "decide"

    def finish_repair(self, state: TurnState) -> TurnState:
        result = state.get("repair")
        if result is None:
            answer = "本轮没有运行补丁测试。"
            if state["answer"]:
                answer += "\n\n" + state["answer"]
            return {"action": None, "answer": answer}
        status = result["status"]
        messages = {
            "passed": "补丁已通过配置的沙箱测试。",
            "failed": "补丁未通过配置的沙箱测试。",
            "timeout": "沙箱测试超时。",
            "patch_rejected": "最后提交的补丁无法应用。",
            "sandbox_error": "沙箱或测试命令无法运行。",
            "not_reproduced": "原始快照上没有复现指定问题，本轮没有生成补丁。",
            "regression_failed": "回归测试未通过。",
        }
        if result.get("phase") == "baseline_regression" and status == "regression_failed":
            return {"action": None, "answer": "原始快照上的回归测试已经失败，本轮没有生成补丁。"}
        final_result = {**result, "submission_error": state["submission_error"]} if state.get("submission_error") else result
        if status == "passed" and state.get("interactive_turn"):
            try:
                workspace = self._workspace(state)
                apply_verified_patch(workspace, self.agent._snapshot_path(state["turn_id"]),
                                     self.agent.config.state_dir, result["patch"], result["base_digest"])
            except (OSError, ValueError, TimeoutExpired) as exc:
                return {"action": None, "answer": f"补丁已通过测试，暂未应用。{exc}",
                        "repair": {**final_result, "apply_error": str(exc)}}
            added_tests = sum(phase.startswith("new_test_after:") for phase in result.get("verification", {}))
            answer = "修改已通过测试并应用到当前项目。"
            if added_tests:
                answer += f"已新增并通过 {added_tests} 个测试文件。"
            return {"action": None, "answer": answer,
                    "repair": {**final_result, "applied_to": str(workspace)}}
        answer = messages[status]
        if status == "sandbox_error" and result.get("error_kind") in {"no_tests", "missing_dependency", "unsupported_test_runner"}:
            answer = result["output"]
        return {"action": None, "answer": answer, "repair": final_result}

    @staticmethod
    def input(query: str, task_id: str | None, mode: WorkflowMode, compaction: ContextCompactionState) -> TurnState:
        return {"query": query, "task_id": task_id, "mode": mode, "events": [], "observations": [],
                "calls": [], "action": None, "answer": "", "compaction": compaction,
                "repair": None, "tested_patches": [], "validation_runs": 0, "submission_error": None,
                "read_correction_used": False, "submission_only": False, "baseline_evidence": "",
                "baseline": None, "base_commit": None, "context": None, "sandbox_settings": None, "test_plan": None,
                "require_new_tests": False, "stream_answer": False,
                "interactive_turn": mode == "route", "target": ".", "prior_context": None,
                "project_settings": None, "project_files": {}, "project_steps": 0, "project_verify": False,
                "project_run": None, "project_error": "", "project": None}

    def prepare_context(self, state: TurnState) -> TurnState:
        mode = state["mode"]
        self._emit("context_started", "正在准备本轮上下文")
        instructions = self.agent.system_instructions
        events = []
        scoped_terminal = state.get("interactive_turn") and getattr(self.agent, "terminal_memory", None) is not None
        service = None if scoped_terminal else getattr(self.agent.memory_tool, "service", None)
        bridge = getattr(service, "embedding", None)
        if mode == "repair" and state.get("embedding_profile") and bridge is not None:
            try:
                snapshot = service.repository.read_snapshot()
                status = bridge.index.status(bridge.namespace)
                queued_ids = {event.memory_id for event in snapshot.index_events}
                if any(record.id not in queued_ids for record in snapshot.records) and not (
                    status.ready or status.pending or status.running or status.failed
                ):
                    service.rebuild_index()
                    snapshot = service.repository.read_snapshot()
                sync = service.flush_index_events(limit=len(snapshot.index_events))
                status = bridge.index.prepare_namespace(bridge.namespace)
                events.append(TraceEvent("memory_index", "已处理记忆索引任务", {
                    "ready": status.ready, "pending": status.pending, "running": status.running,
                    "failed": status.failed, "unsynced": sync.pending,
                }))
            except (OSError, RuntimeError) as exc:
                events.append(TraceEvent("memory_index", "记忆索引准备失败", {"reason": type(exc).__name__}))
        notes = []
        if not scoped_terminal:
            try:
                notes = self.agent._notes_to_packets(state["query"])
            except ExercisePending as exc:
                if exc.exercise_id != "E02":
                    raise
                events.append(TraceEvent("note_unavailable", str(exc)))

        if mode not in {"chat", "create", "clarify"}:
            instruction = {
                "explore": "概述仓库结构、主要入口和你实际查看的文件。",
                "issue": "定位问题原因，引用观察到的文件和行号，并指出尚未验证的部分。",
                "plan": "依据证据列出可执行步骤与验收方法。",
                "repair": "定位问题，提交精确的源码修改，并根据沙箱日志迭代。",
            }[mode]
            tools = ("list_files(relative_path, pattern), read_file(relative_path, start_line, end_line), "
                     "search(query, relative_path, limit)")
            if self.agent.rag_tool is not None or mode == "repair":
                tools += ", rag_search(query, limit)"
            if hasattr(self.agent, "terminal_tool"):
                tools += ", find_symbol(symbol, limit)"
            instructions += (
                f"\n{instruction} 可用只读工具：{tools}。"
                '需要补充证据时只输出一个 JSON 对象，例如 '
                '{"action":"search","arguments":{"query":"normalize_query"}}。'
                f"取证工具最多调用 {MAX_TOOL_CALLS} 次。"
            )
            if state.get("target", ".") != ".":
                instructions += (f'\n本轮工具已在目标项目 {state["target"]!r} 的根目录工作。'
                                 '文件路径相对于项目根目录，省略外层目录前缀。')
            if mode == "repair":
                instructions += (
                    '\n根据已提供的堆栈源码或本轮读取结果构造补丁；定位充分时直接提交。'
                    '整个仓库取证时省略 relative_path。仅需要更多相关行时才继续读下一页。'
                    '取证次数与补丁提交、运行测试次数分开。'
                    '提交时整条回复只输出 {"action":"verify_edits","arguments":{"edits":'
                    '[{"path":"相对文件路径","old_text":"唯一匹配的原文","new_text":"替换文本"}]}}。'
                    'old_text 必须逐字对应原始快照，保留缩进和换行，用最短的唯一上下文。'
                    'new_text 是实际修改后的版本，只提交发生变化的替换项。'
                    '每次提交包含本次完整修改，全部相对于原始快照。可用 verify_patch 提交标准统一 diff。'
                    f"最多提交 {MAX_REPAIR_SUBMISSIONS} 次，最多实际验证 {MAX_REPAIR_ATTEMPTS} 次。"
                    "每次修订先核对失败输入、预期结果与实际结果，再检查哪一处代码改变会使该用例通过。"
                    "保留用例未要求改变的行为。测试命令已由操作者配置。"
                    '测试通过时流程会直接返回补丁。确实无法修复时可用 {"answer":"原因"} 结束。'
                )
                if state.get("interactive_turn"):
                    instructions += (
                        '新增文件时，在 verify_edits 中将 old_text 设为空串，并提交完整 new_text。'
                        '新增用例请创建单独的 tests/test_*.py 文件，已有测试受保护。使用 unittest 时用 TestCase 编写。'
                        '需要诊断时可调用 run_tests(file)，省略 file 跑整套。该工具测试最近的候选补丁，'
                        '尚无候选时测试原始源码。最终交付仍由 verify_edits 或 verify_patch 的完整测试决定。'
                    )
                    if state.get("require_new_tests"):
                        instructions += '用户明确要求新增测试。把完整的新测试文件和源码改动放在同一次提交里。'
            elif state.get("stream_answer"):
                instructions += ('取证完成后用 {"answer":"最终答复的要点、依据及必要的文件行号"} 结束。'
                                 '保留足够的信息供后续生成正文。')
            else:
                instructions += '可以用 {"answer":"最终回答"} 或直接用文本结束。'
        packets = [*notes, state["prior_context"]] if state.get("prior_context") else notes
        if scoped_terminal:
            instructions += ("\n历史进度只说明当时的交付和验证。涉及当前代码状态时，按实际证据判断。"
                             "若显式个人偏好冲突，遵循本轮要求及最新的相关偏好。")
            if "history_start" not in state:
                state = {**state, **self.agent.scope_terminal_turn(state["target"], state.get("prior_context"))}
                packets = [*notes, state["prior_context"]] if state.get("prior_context") else notes
            scoped = self.agent.build_terminal_context(state, instructions, packets)
            context = scoped["context"]
        else:
            context = self.agent._build_context_for_turn(
                state["query"], task_id=state["task_id"], custom_packets=packets or None,
                system_instructions=instructions if mode != "chat" else None,
            )
            scoped = {"compaction": self.agent.compaction_state}
        if events:
            context.warnings.extend(event.message for event in events)
        events.append(self._emit("context", "上下文已就绪", tokens=context.token_count))
        if scoped.get("context_budget", {}).get("newly_covered"):
            events.append(self._emit("context_compaction", "已压缩较早的对话", **scoped["context_budget"]))
        prepared = {**scoped, "instructions": instructions, "context": context,
                    "events": [*state["events"], *events]}
        if mode == "explore":
            prepared["action"] = {"action": "list_files", "arguments": {}}
        return prepared

    def decide(self, state: TurnState) -> TurnState:
        instructions = state["instructions"]
        submission_only = state.get("submission_only", False)
        prompt = state["context"].context
        if prompt.startswith("[Role & Policies]\n"):
            _, separator, rest = prompt.partition("\n\n[Task]\n")
            if separator:
                prompt = "[Task]\n" + rest
        repair = state.get("repair")
        baseline = state.get("baseline")
        reads_left, submissions_left, tests_left = self._repair_budget(state)
        if state["mode"] == "repair" and (repair is not None or baseline):
            # 失败反馈直接附到模型输入，避免被相关性筛选丢弃。
            original_failure = baseline["reproduction"] if baseline else None
            candidate_test = repair if repair and repair["status"] != "patch_rejected" else None
            verification = candidate_test or original_failure or repair
            counter = getattr(self.agent.context_builder, "token_counter", None) or ApproxTokenCounter()
            budget = min(2_048, max(1, self.agent.config.context.input_budget // 5))
            report = {key: verification[key] for key in ("status", "exit_code")}
            report["test_command"] = list(self._repair_test_command(state))
            report["phase"] = candidate_test.get("phase", "candidate") if candidate_test else "baseline_reproduction"
            report["output"] = counter.truncate(verification["output"][::-1], budget)[::-1]
            parts = [f"剩余取证 {reads_left} 次，补丁提交 {submissions_left} 次，实际测试 {tests_left} 次。"]
            if submission_only:
                parts.append("本轮请直接提交实质性的补丁，或明确说明缺少哪条证据。")
            if evidence := state.get("baseline_evidence"):
                parts.append("[原始堆栈相邻源码]\n" + counter.truncate(evidence, budget))
            if candidate_test and original_failure and original_failure["status"] == "failed":
                original_log = counter.truncate(original_failure["output"][::-1], max(1, budget // 2))[::-1]
                parts.append("[原始复现失败]\n" + original_log)
            parts.append("[最近沙箱结果]\n" + json.dumps(report, ensure_ascii=False))
            if candidate_test and candidate_test["patch"]:
                parts.append("[最近受测补丁的前缀]\n" + counter.truncate(candidate_test["patch"], budget))
            if state.get("submission_error"):
                text = json.dumps(state["submission_error"], ensure_ascii=False)
                parts.append("[最近提交错误]\n" + counter.truncate(text, budget))
            feedback = "\n\n".join(parts)
            remaining = max(1, self.agent.config.context.input_budget - counter.count(instructions) - counter.count(feedback))
            prompt = counter.truncate(prompt, remaining) + "\n\n" + feedback
        self._emit("model_started", "正在等待模型决定下一步")
        answer = self.agent.model.generate([Message("system", instructions), Message("user", prompt)])
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("回答模型必须返回非空文本")
        # 预算裁剪可能移除整段证据，记录实际出现在模型输入中的引用。
        references = {
            (ref["source"], ref["line"], ref["mode"])
            for packet in state["context"].selected
            for ref in packet.metadata.get("rag_references", ())
            if f"{ref['source']}:{ref['line']}\n" in prompt
        }
        events = [*state["events"], TraceEvent("model", "已获得模型回复", {
            "rag_evidence": [{"source": source, "line": line, "mode": mode}
                             for source, line, mode in sorted(references)]
        })]
        if state["mode"] == "chat":
            return {"answer": answer, "action": None, "events": events}
        payload = answer.strip()
        if payload.startswith("```json\n") and payload.endswith("```"):
            payload = payload[len("```json\n"):-3].strip()
        decision = None
        if payload.startswith("{") or (state["mode"] == "repair" and payload.startswith("[")):
            try:
                decoded = json.loads(payload)
                if not isinstance(decoded, dict):
                    raise ValueError("取证请求必须是 JSON 对象")
                if set(decoded) == {"answer"} and isinstance(decoded["answer"], str) and decoded["answer"].strip():
                    answer = decoded["answer"]
                elif set(decoded) == {"action", "arguments"} and isinstance(decoded["action"], str):
                    decision = decoded
                else:
                    raise ValueError("模型应返回 action/arguments 或非空的 answer")
            except ValueError:
                if state["mode"] != "repair":
                    raise
                decision = {"action": INVALID_RESPONSE, "arguments": {}}
        if decision is None and state["mode"] == "repair":
            blocks = DIFF_BLOCK.findall(answer)
            if len(blocks) == 1 and blocks[0].startswith(("diff --git ", "--- ")):
                decision = {"action": VERIFY_PATCH, "arguments": {"patch": blocks[0]}}
                events.append(TraceEvent("repair", "从模型回复提取统一 diff 并提交验证"))
        update: TurnState = {"answer": answer, "action": None, "events": events, "submission_only": False}
        if decision is None:
            return update
        if state["mode"] == "repair":
            if decision["action"] in REPAIR_SUBMISSIONS:
                if not (submissions_left and tests_left):
                    return {**update, "answer": "已到本轮补丁验证上限。"}
            elif not reads_left or submission_only:
                has_source = bool(state.get("baseline_evidence")) or any(
                    packet.metadata.get("tool") == "read_file" and packet.metadata.get("status") == "ok"
                    for packet in state["observations"]
                )
                redirect = not submission_only and has_source and bool(submissions_left and tests_left)
                if redirect:
                    events.append(TraceEvent("repair", "取证结束，转入补丁提交"))
                return {**update, "answer": "已到本轮取证上限。", "submission_only": redirect}
        elif len(state["calls"]) >= MAX_TOOL_CALLS:
            raise ValueError("达到工具调用上限，模型仍未提交分析")
        return {"action": decision, "events": events, "submission_only": False}

    def render_answer(self, state: TurnState) -> TurnState:
        """将已确定的公开答复流式显示，只保存正常结束的正文。"""
        stream = getattr(self.agent.model, "generate_stream", None)
        output = self.agent.answer_output
        draft = state["answer"]
        if output is None or not callable(stream):
            return {"answer": draft}

        instructions = (
            "根据已经确定的答复要点和验证事实，用用户语言写出最终答复。"
            "保留关键文件、既有引用和实际测试结论。只输出面向用户的 Markdown 正文。"
            "不要增加未验证的事实，不要重复任务，不要发出工具调用。"
        )
        repair, project = state.get("repair") or {}, state.get("project") or {}
        facts = {
            "已改文件": [item["path"] for item in repair.get("changed_files", [])] or project.get("files", []),
            "测试结果": {name: run["status"] for name, run in repair.get("verification", {}).items()},
        }
        if run := project.get("verification"):
            facts["测试结果"]["项目测试"] = run["status"]
        counter = getattr(self.agent.context_builder, "token_counter", None) or ApproxTokenCounter()
        room = max(1, self.agent.config.context.input_budget - counter.count(instructions) - 16)
        request = counter.truncate(state["query"], max(1, room // 4))
        data = f"请求\n{request}\n\n答复要点\n{draft}"
        if facts["已改文件"] or facts["测试结果"]:
            data += "\n\n验证事实\n" + json.dumps(facts, ensure_ascii=False)
        messages = [Message("system", instructions), Message("user", counter.truncate(data, room))]
        self._emit("stream_started", "正在生成最终答复")
        started = perf_counter()
        first_ms = None
        parts = []
        try:
            with closing(stream(messages)) as chunks:
                for part in chunks:
                    if not part:
                        continue
                    if first_ms is None:
                        first_ms = round((perf_counter() - started) * 1_000)
                    parts.append(part)
                    output(part)
        finally:
            output(None)
        answer = "".join(parts)
        if not answer.strip():
            raise RuntimeError("模型流未返回正文，可用 --no-stream 和 /resume 显示已保存的答复要点")
        event = TraceEvent("model", "已完成答复输出", {"streamed": True, "chunks": len(parts), "first_delta_ms": first_ms})
        return {"answer": answer, "events": [*state["events"], event]}

    @staticmethod
    def _repair_budget(state: TurnState) -> tuple[int, int, int]:
        submissions = sum(call["action"] in REPAIR_SUBMISSIONS for call in state["calls"])
        reads = len(state["calls"]) - submissions - state.get("read_correction_used", False)
        return (MAX_TOOL_CALLS - reads, MAX_REPAIR_SUBMISSIONS - submissions,
                MAX_REPAIR_ATTEMPTS - state.get("validation_runs", 0))

    def _read_traceback_source(self, snapshot: Path, output: str) -> str:
        """从复现日志映射最内层仓库帧，只在已冻结的源码快照中读取。"""
        counter = getattr(self.agent.context_builder, "token_counter", None) or ApproxTokenCounter()
        budget = min(1_600, max(1, self.agent.config.context.input_budget // 5))
        terminal = TerminalTool(snapshot, self.agent.config.terminal_max_chars)
        for name, line_text in reversed(TRACE_FRAME.findall(output)):
            path = Path(name)
            if path.is_relative_to("/work"):
                path = path.relative_to("/work")
            elif path.is_absolute():
                parts = path.parts
                for index, part in enumerate(parts[:-2]):
                    if part.startswith("coding-agent-baseline-") and parts[index + 1] == "candidate":
                        path = Path(*parts[index + 2:])
                        break
            if path.is_absolute() or path.suffix != ".py" or "tests" in path.parts or "test" in path.parts:
                continue
            line = int(line_text)
            try:
                page = terminal.read_file_page(
                    path.as_posix(), max(1, line - 38), line + 16, max_tokens=budget, count_tokens=counter.count,
                )
            except (OSError, ValueError):
                continue
            if (page["end_line"] or 0) >= line:
                return f"{path}:{page['start_line']}-{page['end_line']}（堆栈行 {line}）\n{page['text']}"
        return ""

    def execute_tool(self, state: TurnState) -> TurnState:
        action = state["action"]
        assert action is not None
        tool, arguments = action["action"], action["arguments"]
        if state["mode"] == "repair" and tool in REPAIR_SUBMISSIONS:
            return self._verify_patch(state, arguments, tool)
        self._emit("tool_started", "正在运行诊断测试" if tool == RUN_TESTS else "正在取证", tool=tool)
        builder = self.agent.context_builder
        counter = getattr(builder, "token_counter", None) or ApproxTokenCounter()
        max_tokens = min(DEFAULT_EVIDENCE_TOKENS, max(1, self.agent.config.context.input_budget // (2 * MAX_TOOL_CALLS)))
        structured_read = False
        location = None
        references = []
        read_correction_used = state.get("read_correction_used", False)
        try:
            if (tool in {"read_file", "search", "list_files"} and isinstance(arguments, dict)
                    and "path" in arguments and "relative_path" not in arguments):
                arguments = {key: value for key, value in arguments.items() if key != "path"}
                arguments["relative_path"] = action["arguments"]["path"]
                action = {"action": tool, "arguments": arguments}
            if isinstance(arguments, dict) and isinstance(arguments.get("relative_path"), str):
                arguments = {**arguments, "relative_path": self._project_relative(arguments["relative_path"], state)}
                action = {"action": tool, "arguments": arguments}
            if tool not in TOOL_ARGUMENTS:
                raise ToolArgumentsError("未知的只读工具")
            if tool != RUN_TESTS and action in state["calls"]:
                raise ValueError("该工具和参数已使用过，请换一个取证动作或提交分析")
            data = self._read_evidence(tool, arguments, state, max_tokens, counter)
            if tool == "read_file":
                location = f"{data['path']}:{data['start_line']}"
            if tool == "read_file" and "next_start_line" in data:
                header = {key: value for key, value in data.items() if key != "text"}
                header["relative_path"] = header.pop("path")
                observation = json.dumps(header, ensure_ascii=False) + "\n" + data["text"]
                structured_read = True
            elif tool == "rag_search" and state["mode"] == "repair":
                budget = max_tokens - 64
                blocks = []
                for item in data:
                    block = f"{item['source']}:{item['line']}\n{item['text']}"
                    cost = counter.count(block)
                    if cost <= budget:
                        blocks.append(block)
                        references.append({"source": item["source"], "line": item["line"],
                                           "mode": state.get("rag_mode", "lexical")})
                        budget -= cost
                observation = "\n\n".join(blocks) or "[]"
            else:
                observation = json.dumps(data, ensure_ascii=False)
            status = "ok"
        except (OSError, ValueError, RuntimeError) as exc:
            observation = (exc.strerror or "文件操作失败") if isinstance(exc, OSError) else str(exc)
            status = type(exc).__name__
            if state["mode"] == "repair" and isinstance(exc, ToolArgumentsError) and not read_correction_used:
                read_correction_used = True
                observation += "。本次可订正参数，不占用取证额度"

        tool_name = tool if tool in TOOL_ARGUMENTS else "未知工具"
        remaining = MAX_TOOL_CALLS - sum(call["action"] not in REPAIR_SUBMISSIONS for call in state["calls"]) - 1
        request = json.dumps(arguments, ensure_ascii=False) if status == "ok" and not structured_read else ""
        if request:
            request = " " + counter.truncate(request, max(1, max_tokens // 4))
        progress = f"剩余工具次数 {remaining}\n" if state["mode"] != "repair" else ""
        packet_text = f"{progress}工具 {tool_name}（{status}）{request}\n{observation}"
        if structured_read and counter.count(packet_text) > max_tokens:
            status = "ValueError"
            packet_text = f"工具 read_file（{status}）\n页头开销超出预算，请缩小行范围"
        elif not structured_read:
            packet_text = counter.truncate(packet_text, max_tokens)
        packet = ContextPacket(packet_text, relevance_score=1.0,
                               metadata={"type": "rag_result", "tool": tool_name, "status": status,
                                         "rag_references": references})
        observations = [*state["observations"], packet]
        context = state["context"]
        source_packets = [item for item in context.candidates
                          if item.metadata.get("type") not in {"system_instruction", "conversation_history"}
                          and "tool" not in item.metadata]
        compaction = state["compaction"]
        history = self.agent.history[compaction.covered_until:]
        if "history_start" in state:
            history = [Message(item.metadata["role"], item.content, item.timestamp) for item in context.candidates
                       if item.metadata.get("type") == "conversation_history"]
        refreshed = builder.build(
            state["query"], conversation_history=history,
            system_instructions=state["instructions"], custom_packets=deepcopy(source_packets + observations),
            sources_prepared=True,
        )
        refreshed.warnings[:0] = context.warnings
        if status != "ok":
            refreshed.warnings.append(f"{tool_name} 未返回证据（{status}）")
        details = {"tool": tool_name, "status": status}
        if location is not None:
            details["location"] = location
        if tool_name == "rag_search" and state["mode"] == "repair":
            details.update(mode=state.get("rag_mode", "lexical"), hits=references)
        if tool == RUN_TESTS and status == "ok":
            details.update(status=data["status"], error_kind=data.get("error_kind"),
                           elapsed_seconds=data.get("elapsed_seconds"))
        event = self._emit("tool", "诊断测试完成" if tool == RUN_TESTS else "取证完成", **details)
        return {"context": refreshed, "observations": observations, "calls": [*state["calls"], action],
                "events": [*state["events"], event], "read_correction_used": read_correction_used}

    def _verify_patch(self, state: TurnState, arguments: Any, tool: str = VERIFY_PATCH) -> TurnState:
        tested = list(state.get("tested_patches", []))
        snapshot = self.agent._snapshot_path(state["turn_id"])
        previous = state.get("repair")
        runs = state.get("validation_runs", 0)
        command = self._repair_test_command(state)
        allow_new_tests = bool(state.get("interactive_turn") and test_runner(command))
        patch, submission_error = "", None
        attempted = False
        number = 1 + MAX_REPAIR_SUBMISSIONS - self._repair_budget(state)[1]
        self._emit("submission_started", "正在处理补丁提交", submission=number)
        try:
            if tool == INVALID_RESPONSE:
                raise ValueError("请提交 JSON 对象，包含合法的 action 和 arguments")
            if tool == VERIFY_EDITS:
                if not isinstance(arguments, dict) or set(arguments) != {"edits"}:
                    raise ValueError("verify_edits 需要提供 edits 列表")
                if state.get("interactive_turn"):
                    edits = arguments["edits"]
                    if isinstance(edits, list):
                        edits = [{**edit, "path": self._project_relative(edit["path"], state)}
                                 if isinstance(edit, dict) and isinstance(edit.get("path"), str) else edit
                                 for edit in edits]
                    patch = self.agent.sandbox.patch_from_edits(snapshot, edits, allow_create=True,
                                                               allow_new_tests=allow_new_tests)
                else:
                    patch = self.agent.sandbox.patch_from_edits(snapshot, arguments["edits"])
            elif isinstance(arguments, dict) and set(arguments) == {"patch"} and isinstance(arguments["patch"], str):
                patch = arguments["patch"]
            if not patch.strip() or len(patch) > MAX_PATCH_CHARS or "\x00" in patch:
                raise ValueError("需要提供非空且长度合法的修改")
            if not patch.endswith("\n"):
                patch += "\n"
            if patch in tested:
                submission_error = {"kind": "duplicate", "message": "相同补丁已处理，请根据此前验证结果提交不同的修改"}
                result = {"status": "patch_rejected", "patch": patch, "exit_code": None,
                          "output": submission_error["message"]}
            else:
                tested.append(patch)
                attempted = True
                self._emit("verification_started", "正在验证候选补丁", submission=number, test=runs + 1)
                kwargs = {"command": command} if command != self.agent.sandbox.test_command else {}
                if allow_new_tests:
                    kwargs["allow_new_tests"] = True
                if state.get("require_new_tests"):
                    kwargs["require_new_tests"] = True
                result = self.agent.sandbox.verify(snapshot, patch, **kwargs)
                if result["patch"] and result["patch"] != patch:
                    tested.append(result["patch"])
                if result["status"] == "passed":
                    result = {**result, "base_digest": tree_digest(snapshot), "base_commit": state.get("base_commit")}
                elif result["status"] == "patch_rejected":
                    submission_error = {"kind": result.get("error_kind", "patch_rejected"), "message": result["output"]}
        except ValueError as exc:
            submission_error = {"kind": "invalid_response" if tool == INVALID_RESPONSE else "invalid_edit", "message": str(exc)}
            result = {"status": "patch_rejected", "patch": "", "exit_code": None, "output": str(exc)}
        except (OSError, RuntimeError, TimeoutExpired) as exc:
            result = {"status": "sandbox_error", "patch": patch, "exit_code": None,
                      "output": f"准备沙箱失败（{type(exc).__name__}）"}

        ran_tests = attempted and result["status"] in {"passed", "failed", "timeout", "regression_failed"}
        event = self._emit("tool", "补丁提交处理完成", tool=tool, status=result["status"],
                           exit_code=result["exit_code"], ran_tests=ran_tests, submission=number,
                           verification={phase: run["status"] for phase, run in result.get("verification", {}).items()},
                           patch_recounted=result.get("patch_recounted", False), submission_error=submission_error)
        if submission_error and previous and previous["status"] in {"failed", "timeout", "regression_failed"}:
            result = previous
        elif submission_error and submission_error["kind"] == "duplicate" and previous and previous["patch"] == patch:
            result = previous
        return {"repair": result, "tested_patches": tested, "validation_runs": runs + int(ran_tests),
                "submission_error": submission_error, "calls": [*state["calls"], {"action": tool}],
                "events": [*state["events"], event]}

    def _repair_symbols(self, snapshot: Path, mode: str, workspace: Path) -> PythonSymbols:
        pattern = self.agent.index_pattern
        if mode == "lexical":
            return PythonSymbols(snapshot, self.agent.config.state_dir, pattern=pattern)
        revision = tree_digest(snapshot)
        if pattern != "**/*.py":
            revision += f":{pattern}"
        rag = RAGTool(
            snapshot, repo_id=str(workspace), revision=revision,
            embedding_index=self.agent.embedding_index, search_options=RAGSearchOptions(mode=mode),
        )
        return PythonSymbols(snapshot, self.agent.config.state_dir, rag=rag, pattern=pattern)

    def _read_evidence(self, tool: str, arguments: Any, state: TurnState, max_tokens: int, counter: Any) -> Any:
        if tool not in TOOL_ARGUMENTS or not isinstance(arguments, dict):
            raise ToolArgumentsError("未知的只读工具或非法参数")
        allowed, required = TOOL_ARGUMENTS[tool]
        if set(arguments) - allowed or required - set(arguments):
            raise ToolArgumentsError(f"工具参数字段不符合接口。可用字段：{', '.join(sorted(allowed))}")
        for name, value in arguments.items():
            if name == "end_line" and value is None:
                continue
            expected = int if name in {"limit", "start_line", "end_line"} else str
            if type(value) is not expected:
                raise ToolArgumentsError(f"工具参数 {name} 类型不符合接口")
        if "limit" in arguments and not 1 <= arguments["limit"] <= 20:
            raise ToolArgumentsError("取证 limit 必须介于 1 和 20 之间")
        if tool == RUN_TESTS:
            if state["mode"] != "repair" or not state.get("interactive_turn"):
                raise ToolArgumentsError("run_tests 仅可在终端修改任务中使用")
            result = state.get("repair") or {}
            patch = result.get("patch", "") if result.get("status") in {"failed", "regression_failed", "timeout"} else ""
            file = arguments.get("file")
            return self.agent.sandbox.run_tests(self.agent._snapshot_path(state["turn_id"]), self._repair_test_command(state),
                                                patch=patch, file=self._project_relative(file, state) if file is not None else None)
        if tool == "find_symbol" or (tool == "rag_search" and state["mode"] == "repair"):
            if self.symbols is None:
                if state["mode"] == "repair":
                    snapshot = self.agent._snapshot_path(state["turn_id"])
                    self.symbols = self._repair_symbols(snapshot, state.get("rag_mode", "lexical"), self._workspace(state))
                else:
                    self.symbols = PythonSymbols(self._workspace(state), self.agent.config.state_dir)
            if tool == "find_symbol":
                return self.symbols.find(**arguments)
        if tool == "rag_search":
            rag = self.symbols.rag if state["mode"] == "repair" else self.agent.rag_tool
            if rag is None:
                raise ValueError("当前未启用 RAG")
            if state["mode"] == "repair":
                options = replace(
                    rag.search_options, mode=state.get("rag_mode", "lexical"),
                    expand_parents=True, max_context_tokens=max(1, max_tokens - 96),
                )
                hits = rag.search(**arguments, options=options)
            else:
                hits = rag.search(**arguments)
            return [{"source": h.chunk.source, "line": h.chunk.metadata.get("start_line") or 1,
                     "text": (h.context_chunk or h.chunk).content} for h in hits]
        terminal = self.agent.terminal_tool
        if state["mode"] == "repair":
            terminal = TerminalTool(self.agent._snapshot_path(state["turn_id"]), self.agent.config.terminal_max_chars)
        elif state.get("target", ".") != ".":
            terminal = TerminalTool(self._workspace(state), self.agent.config.terminal_max_chars)
        if tool == "read_file" and hasattr(terminal, "read_file_page"):
            overhead = 64 + counter.count(json.dumps(arguments["relative_path"], ensure_ascii=False))
            return terminal.read_file_page(**arguments, max_tokens=max(1, max_tokens - overhead), count_tokens=counter.count)
        result = getattr(terminal, tool)(**arguments)
        if tool == "read_file":
            return {"path": arguments["relative_path"], "start_line": arguments.get("start_line", 1), "text": result}
        return result
