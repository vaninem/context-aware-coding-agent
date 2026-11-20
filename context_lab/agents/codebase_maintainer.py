"""通过 LangGraph 获取仓库证据、验证补丁并恢复进行中的回合。"""

import fcntl
import json
import sqlite3
from collections.abc import Callable
from contextlib import closing
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from shutil import rmtree
from sys import maxsize
from typing import TYPE_CHECKING
from uuid import uuid4

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

from ..config import AppConfig
from ..memory.errors import MemorySummaryStale
from ..ports import ChatModel
from ..storage.json_store import JsonFileStore
from ..tokens import ApproxTokenCounter
from ..tools.delivery import base_commit
from ..tools.project_files import project_path
from ..tools.sandbox import DEFAULT_PROJECT_TEST_COMMAND, DockerSandbox, copy_source
from ..tools.terminal import TerminalTool
from ..types import (
    AgentReply, ContextBuildResult, ContextCompactionState, ContextPacket, Message, TraceEvent,
    history_prefix_hash, utc_now,
)
from .project_assistant import ProjectAssistant
from .terminal_memory import TerminalMemory
from .workflow import TurnState, TurnWorkflow, WorkflowMode

if TYPE_CHECKING:
    from ..embeddings import IncrementalEmbeddingIndex


class CodebaseMaintainer(ProjectAssistant):
    def __init__(
        self,
        config: AppConfig,
        model: ChatModel,
        terminal_tool: TerminalTool | None = None,
        *,
        sandbox: DockerSandbox | None = None,
        project_sandbox: DockerSandbox | None = None,
        embedding_index: "IncrementalEmbeddingIndex | None" = None,
        repair_rag_mode: str = "lexical",
        index_pattern: str = "**/*.py",
        progress: Callable[[TraceEvent], None] | None = None,
        carry_context: bool = False,
        automatic_tests: bool = False,
        terminal_memory_dir: Path | None = None,
        **dependencies,
    ) -> None:
        super().__init__(config, model, **dependencies)
        self.terminal_tool = terminal_tool if terminal_tool is not None else TerminalTool(
            config.workspace, max_output_chars=config.terminal_max_chars
        )
        self.sandbox = sandbox
        if project_sandbox is None and sandbox is not None:
            project_sandbox = DockerSandbox(sandbox.image, DEFAULT_PROJECT_TEST_COMMAND, sandbox.timeout_seconds)
        self.project_sandbox = project_sandbox
        if repair_rag_mode not in {"lexical", "vector", "hybrid"}:
            raise ValueError("不支持的修复检索模式")
        if repair_rag_mode != "lexical" and embedding_index is None:
            raise ValueError("修复的向量检索需要配置嵌入索引")
        if not index_pattern.strip() or Path(index_pattern).is_absolute() or ".." in Path(index_pattern).parts:
            raise ValueError("索引范围必须是仓库内的相对 glob")
        self.embedding_index = embedding_index
        self.repair_rag_mode = repair_rag_mode
        self.index_pattern = index_pattern
        self.progress = progress
        self.answer_output: Callable[[str | None], None] | None = None
        self.carry_context = carry_context
        self.automatic_tests = automatic_tests
        self.terminal_memory = None
        if terminal_memory_dir is not None:
            self.terminal_memory = TerminalMemory(terminal_memory_dir, config.user_id, model,
                                                 self.context_builder.token_counter)
        self._reset_session()

    def _reset_session(self, project: str = ".") -> None:
        self.session_id = f"session_{uuid4().hex}"
        self.active_project = project
        self.history_project = project
        self.history_start = 0
        self.completed_turn_id: str | None = None
        self.last_validation: dict | None = None
        self.stats = {
            "session_started_at": utc_now().isoformat(),
            "model_calls": 0,
            "tool_calls": 0,
            "notes_created": 0,
        }

    def start_new_session(self) -> None:
        """清空对话视图，继续使用当前项目的长期记忆。"""
        if self.session_status()["pending"]:
            raise ValueError("请先用 /resume 完成当前回合，再开启新会话")
        self.clear_history()
        self.trace.clear()
        self._reset_session(self.active_project)

    def clear_history(self) -> None:
        super().clear_history()
        self.history_start = 0
        self.history_project = self.active_project

    def explore(self) -> AgentReply:
        """从工作区目录开始取证，按需要读取文件。"""
        return self._run_turn("请探索当前代码库并概述主要结构。", mode="explore")

    def analyze_issue(self, issue: str) -> AgentReply:
        """按问题收集仓库证据并生成分析。"""
        return self._run_turn(issue, mode="issue")

    def plan_task(self, task: str) -> AgentReply:
        """生成有据可查的行动计划并记录为项目笔记。"""
        return self._run_turn(task, mode="plan")

    def repair(self, issue: str) -> AgentReply:
        """生成、验证并返回补丁，测试反馈驱动后续尝试。"""
        return self._run_turn(issue, mode="repair")

    def converse(self, request: str) -> AgentReply:
        """在同一个可恢复回合内识别意图，随后分析、修改或生成项目。"""
        return self._run_turn(request, mode="route")

    def _run_turn(
        self, user_input: str, *, task_id: str | None = None, mode: WorkflowMode = "chat",
    ) -> AgentReply:
        if not isinstance(user_input, str) or not user_input.strip():
            raise ValueError("user_input 必须为非空字符串")
        task_id = task_id if task_id is not None else self.compaction_state.task_id
        if task_id is None and self.config.context.auto_summary_enabled:
            task_id = self.session_id
        state = TurnWorkflow.input(user_input, task_id, mode, self.compaction_state)
        state["turn_id"] = uuid4().hex
        return self._execute_turn(state)

    def resume(self) -> AgentReply:
        """加载已保存会话，从 SQLite 恢复尚未完成的回合。"""
        return self._execute_turn(None)

    def scope_terminal_turn(self, target: str, prior_context: ContextPacket | None) -> TurnState:
        """在路由确定目标后固定本轮历史起点与摘要引用。"""
        if target != self.history_project:
            return {"history_start": len(self.history), "prior_context": None,
                    "compaction": ContextCompactionState(task_id=self.session_id)}
        compaction = self.compaction_state
        if compaction.task_id is None:
            compaction = replace(compaction, task_id=self.session_id)
        return {"history_start": self.history_start, "prior_context": prior_context, "compaction": compaction}

    def build_terminal_context(self, state: TurnState, instructions: str, packets: list[ContextPacket]) -> TurnState:
        """为目标项目召回进度并缩短本段会话的模型输入。"""
        project = project_path(self.config.workspace, state["target"])
        history_start = state["history_start"]
        history = self.history[history_start:]
        query = state["query"]
        config = self.config.context
        builder = self.context_builder
        warnings: list[str] = []
        tool = self.terminal_memory.project_tool(project)
        try:
            packets = packets + self.terminal_memory.recall(tool.service, query, min(900, config.input_budget // 6))
        except (OSError, ValueError, RuntimeError) as exc:
            warnings.append(f"历史记忆读取失败（{type(exc).__name__}）")
        packets = builder.collect_sources(query, packets, warnings=warnings, include_memory=False)
        initial = state["compaction"]
        saved_compaction = self.compaction_state
        self.compaction_state = initial
        fallback = None
        try:
            if config.auto_summary_enabled:
                try:
                    context = self._build_with_summary(
                        query, task_id=initial.task_id, custom_packets=packets, system_instructions=instructions,
                        history=history, summary_tool=tool, sources_prepared=True,
                    )
                except (OSError, ValueError, RuntimeError) as exc:
                    fallback = type(exc).__name__
                    warnings.append(f"阶段摘要暂不可用（{fallback}），本轮使用预算内的最近对话")
                    if isinstance(exc, MemorySummaryStale):
                        self.compaction_state = ContextCompactionState(task_id=initial.task_id)
                    keep = config.summary_keep_recent_messages // 2 * 2
                    recent = history[-keep:] if keep else []
                    while recent and builder.estimate_input(query, recent, instructions) > config.input_budget:
                        recent = recent[2:]
                    context = builder.build(query, recent, instructions, packets, sources_prepared=True)
            else:
                context = builder.build(query, history, instructions, packets, sources_prepared=True)
            compaction = self.compaction_state
        finally:
            self.compaction_state = saved_compaction
        context.warnings[:0] = warnings
        return {"history_start": history_start, "context": context, "compaction": compaction,
                "context_budget": {"input_budget": config.input_budget, "trigger_tokens": config.summary_trigger_tokens,
                                   "covered_messages": compaction.covered_until, "summary_id": compaction.summary_id,
                                   "newly_covered": max(0, compaction.covered_until - initial.covered_until),
                                   "fallback": fallback}}

    def session_status(self) -> dict:
        """查看当前会话已保存的阶段和最近候选验证结果。"""
        status = {"pending": False, "next": (), "turns": len(self.history) // 2,
                  "workspace": str(self.config.workspace / self.active_project),
                  "history": [{"role": item.role, "text": item.content[:400]} for item in self.history[-4:]]}
        if not self.config.workflow_checkpoint_path.is_file():
            return status
        with closing(sqlite3.connect(self.config.workflow_checkpoint_path, check_same_thread=False)) as connection:
            snapshot = self._compile_graph(connection).get_state({"configurable": {"thread_id": self.session_id}})
        state = snapshot.values
        turn_id = state.get("turn_id")
        validation = next((event.metadata for event in reversed(state.get("events", []))
                           if event.metadata.get("ran_tests")), self.last_validation)
        context = state.get("context")
        preview = None
        if context:
            _, marker, body = context.context.partition("\n\n[Task]\n")
            rendered = "[Task]\n" + body if marker else context.context
            preview = {"tokens": context.token_count, "preview": rendered[:1400]}
        project = state.get("project") or {}
        return {**status, "pending": bool(turn_id and turn_id != self.completed_turn_id),
                "next": snapshot.next, "mode": state.get("mode"), "validation": validation,
                "request": state.get("query", "")[:400], "target": state.get("target"),
                "context": preview,
                "project": project.get("status"), "draft_files": sorted(state.get("project_files", {})),
                "result": (state.get("repair") or {}).get("status"), "test_plan": state.get("test_plan"),
                "context_budget": state.get("context_budget")}

    def _compile_graph(self, connection):
        serde = JsonPlusSerializer(allowed_msgpack_modules=[
            ContextBuildResult, ContextCompactionState, ContextPacket, TraceEvent,
        ])
        return TurnWorkflow(self).compile(SqliteSaver(connection, serde=serde))

    def _execute_turn(self, state: TurnState | None) -> AgentReply:
        path = self.config.workflow_checkpoint_path
        path.parent.mkdir(parents=True, exist_ok=True)
        # 一份 state_dir 只运行一个回合，避免并发覆盖 checkpoint.json 的会话入口。
        with path.with_suffix(".lock").open("a") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("当前状态目录已有回合正在执行") from exc
            return self._execute_locked(state)

    def _execute_locked(self, state: TurnState | None) -> AgentReply:
        if state is None:
            self.load_checkpoint()
        thread = {"configurable": {"thread_id": self.session_id}, "recursion_limit": 40}
        with closing(sqlite3.connect(self.config.workflow_checkpoint_path, check_same_thread=False)) as connection:
            graph = self._compile_graph(connection)
            previous = graph.get_state(thread)
            previous_id = previous.values.get("turn_id")
            pending = previous_id is not None and previous_id != self.completed_turn_id
            if state is None:
                if not pending:
                    raise ValueError("当前会话没有待恢复的回合")
                needs_tools = previous.next and previous.next != ("render_answer",)
                if previous.values["mode"] == "repair" and needs_tools and previous.values.get("sandbox_settings"):
                    snapshot = self._snapshot_path(previous_id)
                    if not snapshot.is_dir():
                        raise RuntimeError("找不到本轮修复的仓库快照")
                    if self.sandbox is None or self.sandbox.settings() != previous.values["sandbox_settings"]:
                        raise ValueError("恢复修复回合时需要使用原来的镜像、测试命令和超时")
                    if (
                        previous.values.get("requested_rag_mode", "lexical") != self.repair_rag_mode
                        or previous.values.get("index_pattern", "**/*.py") != self.index_pattern
                        or previous.values.get("embedding_profile") != (
                            self.embedding_index.profile_id if self.embedding_index is not None else None
                        )
                    ):
                        raise ValueError("恢复修复回合时需要使用原来的检索模式、范围和嵌入模型配置")
                    if self.sandbox.protected_tests and not self.sandbox.test_snapshot(snapshot).is_dir():
                        raise RuntimeError("找不到本轮独立测试快照")
                    plan = previous.values.get("test_plan")
                    if previous.next not in {("discover_tests",), ("check_baseline",)} and (plan is None or plan["command"]):
                        self.sandbox.ensure_ready()
                if previous.values["mode"] == "create" and needs_tools and previous.values.get("project_settings"):
                    if self.project_sandbox is None or self.project_sandbox.settings() != previous.values["project_settings"]:
                        raise ValueError("恢复生成回合时需要使用原来的镜像和项目验收命令")
                compaction = previous.values.get("compaction")
                if compaction is not None and self.terminal_memory is None:
                    self.compaction_state = compaction
            else:
                if pending:
                    raise ValueError("当前会话有待恢复的回合，请先调用 resume()")
                latest = JsonFileStore(self.config.checkpoint_path).read()
                if latest is not None:
                    self._validate_checkpoint_owner(latest)
                    if latest.get("session_id") == self.session_id:
                        if latest.get("completed_turn_id") != self.completed_turn_id:
                            raise ValueError("当前实例的会话状态已过期，请先加载最新检查点")
                    elif latest.get("session_id"):
                        saved = graph.get_state({"configurable": {"thread_id": latest["session_id"]}})
                        saved_id = saved.values.get("turn_id")
                        if saved_id is not None and saved_id != latest.get("completed_turn_id"):
                            raise ValueError("状态目录有待恢复的回合，请先调用 resume()")
                # 在图开始之前保存会话入口，进程重启后可找到同一个 thread_id。
                if self.carry_context and len(self.history) > self.history_start:
                    counter = getattr(self.context_builder, "token_counter", None) or ApproxTokenCounter()
                    hint_budget = min(1024, max(1, self.config.context.input_budget // 5))
                    answer = counter.truncate(self.history[-1].content, max(1, hint_budget // 2))
                    observations = "\n\n".join(packet.content for packet in previous.values.get("observations", [])[-2:]
                                                if packet.metadata.get("status") == "ok")
                    hint = counter.truncate(f"{answer}\n\n{observations}", hint_budget)
                    state["prior_context"] = ContextPacket(
                        "上一轮的回答与取证，仅作线索。涉及修改时请在本轮快照重新核对。\n" + hint,
                        relevance_score=1.0, metadata={"type": "previous_context"},
                    )
                if state["mode"] == "repair":
                    state.update(self._freeze_repair(state["turn_id"], self.config.workspace))
                    try:
                        self.save_checkpoint()
                    except BaseException:
                        snapshot = self._snapshot_path(state["turn_id"])
                        rmtree(snapshot, ignore_errors=True)
                        rmtree(DockerSandbox.test_snapshot(snapshot), ignore_errors=True)
                        raise
                else:
                    self.save_checkpoint()

            start = len(self.trace)
            try:
                if state is None and not previous.next:
                    result = previous.values
                else:
                    result = graph.invoke(state, thread, durability="sync")
            except Exception as exc:
                self.trace.append(TraceEvent("error", "本轮未生成回答", {"type": type(exc).__name__}))
                raise

        history_length, prior_stats, prior_turn = len(self.history), self.stats.copy(), self.completed_turn_id
        prior_validation, prior_project = self.last_validation, self.active_project
        prior_history = self.history_project, self.history_start, self.compaction_state
        scoped = result.get("interactive_turn") and self.terminal_memory is not None
        try:
            if result["mode"] == "plan":
                tag = f"plan:{result['turn_id']}"
                if not any(tag in note.tags for note in self.note_tool.list_notes(limit=maxsize, note_type="action")):
                    self.record_note("任务计划", result["answer"], note_type="action", tags=[tag])
                self.stats["notes_created"] += 1
            user_message = self._record_turn(result["query"], result)
            self.stats["model_calls"] += sum(event.stage == "model" for event in result["events"])
            self.stats["tool_calls"] += sum(event.stage == "tool" for event in result["events"])
            self.completed_turn_id = result["turn_id"]
            self.last_validation = next((event.metadata for event in reversed(result["events"])
                                         if event.metadata.get("ran_tests")), self.last_validation)
            if result.get("interactive_turn") and (result["mode"] in {"issue", "repair"}
                    or (result.get("project") or {}).get("status") == "generated"):
                self.active_project = result["target"]
            if scoped:
                self.history_project = result["target"]
                self.history_start = result.get("history_start", history_length)
                self.compaction_state = (result["compaction"] if "history_start" in result
                                         else ContextCompactionState(task_id=self.session_id))
                try:
                    event = self.terminal_memory.save_turn(
                        project_path(self.config.workspace, result["target"]), result, self.session_id,
                    )
                except (OSError, ValueError, RuntimeError) as exc:
                    event = TraceEvent("memory_error", "终端记忆保存失败", {"type": type(exc).__name__})
                if event is not None:
                    self.trace.append(event)
                    if self.progress is not None:
                        self.progress(event)
            self.save_checkpoint()
        except BaseException:
            del self.history[history_length:]
            del self.trace[start:]
            self.stats = prior_stats
            self.completed_turn_id = prior_turn
            self.last_validation = prior_validation
            self.active_project = prior_project
            self.history_project, self.history_start, self.compaction_state = prior_history
            raise
        if not scoped:
            self._write_turn_memory(user_message)
        if result["mode"] == "repair":
            snapshot = self._snapshot_path(result["turn_id"])
            rmtree(snapshot, ignore_errors=True)
            rmtree(DockerSandbox.test_snapshot(snapshot), ignore_errors=True)
        return AgentReply(result["answer"], result["context"], self.trace[start:], result.get("repair"),
                          result.get("baseline"), result.get("project"))

    def _freeze_repair(self, turn_id: str, workspace: Path, *, defer_sandbox: bool = False) -> TurnState:
        if self.sandbox is None:
            raise ValueError("修改项目需要配置 Docker 沙箱")
        snapshot = self._snapshot_path(turn_id)
        state_dir = self.config.state_dir
        if workspace.is_relative_to(state_dir):
            raise ValueError("状态目录必须和目标源码分开")
        if not defer_sandbox:
            self.sandbox.ensure_ready()
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        try:
            # 恢复到这个节点时，前一次可能只复制了一部分。
            rmtree(snapshot, ignore_errors=True)
            rmtree(self.sandbox.test_snapshot(snapshot), ignore_errors=True)
            copy_source(workspace, snapshot, state_dir)
            self.sandbox.freeze_tests(snapshot, workspace, state_dir)
        except BaseException:
            rmtree(snapshot, ignore_errors=True)
            rmtree(self.sandbox.test_snapshot(snapshot), ignore_errors=True)
            raise
        return {"sandbox_settings": self.sandbox.settings(), "requested_rag_mode": self.repair_rag_mode,
                "index_pattern": self.index_pattern, "base_commit": base_commit(workspace),
                "embedding_profile": self.embedding_index.profile_id if self.embedding_index is not None else None}

    def _snapshot_path(self, turn_id: str) -> Path:
        return self.config.state_dir / "repair-snapshots" / turn_id

    def save_checkpoint(self) -> Path:
        """原子保存会话、完整历史、统计及阶段摘要的引用。"""
        path = self.config.checkpoint_path
        history_prefix_hash(self.history)
        JsonFileStore(path).write({
            "schema_version": 1, "project_name": self.config.project_name,
            "workspace": str(self.config.workspace), "user_id": self.config.user_id,
            "session_id": self.session_id, "completed_turn_id": self.completed_turn_id,
            "last_validation": self.last_validation,
            "active_project": self.active_project,
            **({"history_project": self.history_project, "history_start": self.history_start}
               if self.terminal_memory is not None else {}),
            "history": [{**asdict(message), "timestamp": message.timestamp.isoformat()} for message in self.history],
            "stats": self.stats, "compaction_state": asdict(self.compaction_state),
        })
        return path

    def load_checkpoint(self) -> dict:
        """校验快照身份与摘要前缀，成功后整体恢复会话。"""
        path = self.config.checkpoint_path
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        self._validate_checkpoint_owner(data)
        try:
            history = [Message(row["role"], row["content"], datetime.fromisoformat(row["timestamp"]))
                       for row in data["history"]]
            history_prefix_hash(history)
            compaction = ContextCompactionState(**data.get("compaction_state", {}))
            session_id = data["session_id"]
            stats = data["stats"]
            completed_turn_id = data.get("completed_turn_id")
            last_validation = data.get("last_validation")
            active_project = data.get("active_project", ".")
            project_path(self.config.workspace, active_project)
            history_project = data.get("history_project", active_project)
            project_path(self.config.workspace, history_project)
            history_start = data.get("history_start", 0)
            if type(history_start) is not int or not 0 <= history_start <= len(history) or history_start % 2:
                raise ValueError("项目历史起点无效")
            if not isinstance(session_id, str) or not session_id.strip() or not isinstance(stats, dict):
                raise ValueError("会话信息无效")
            if completed_turn_id is not None and not isinstance(completed_turn_id, str):
                raise ValueError("回合 ID 无效")
            if last_validation is not None and (not isinstance(last_validation, dict)
                                                or not isinstance(last_validation.get("status"), str)):
                raise ValueError("最近验证记录格式无效")
            if any(type(stats[key]) is not int or stats[key] < 0 for key in ("model_calls", "tool_calls", "notes_created")):
                raise ValueError("统计数据无效")
            started = datetime.fromisoformat(stats["session_started_at"])
            if started.utcoffset() is None:
                raise ValueError("会话时间必须带时区")
            end = compaction.covered_until
            if type(end) is not int or not 0 <= end <= len(history) - history_start:
                raise ValueError("摘要覆盖范围无效")
            if not isinstance(compaction.history_id, str) or not compaction.history_id.strip():
                raise ValueError("摘要历史 ID 无效")
            if compaction.task_id is not None and (not isinstance(compaction.task_id, str) or not compaction.task_id.strip()):
                raise ValueError("摘要任务 ID 无效")
            if end and (not compaction.task_id or not isinstance(compaction.summary_id, str) or not compaction.summary_id
                        or history_prefix_hash(history[history_start:history_start + end]) != compaction.prefix_hash):
                raise ValueError("阶段摘要与历史前缀不一致")
            if end == 0 and (compaction.summary_id is not None or compaction.prefix_hash is not None):
                raise ValueError("摘要覆盖范围与引用不一致")
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError("会话检查点数据损坏") from exc
        self.history = history
        if self.terminal_memory is not None and "history_project" not in data:
            history_start = len(history)
            compaction = ContextCompactionState()
        self.compaction_state = compaction
        self.session_id = session_id
        self.completed_turn_id = completed_turn_id
        self.last_validation = last_validation
        self.active_project = active_project
        self.history_project, self.history_start = history_project, history_start
        self.stats = stats.copy()
        self.last_context_usage = None
        self.trace.clear()
        return data

    def _validate_checkpoint_owner(self, data: object) -> None:
        if not isinstance(data, dict) or type(data.get("schema_version")) is not int or data["schema_version"] != 1:
            raise ValueError("不支持的会话检查点版本")
        if (data.get("project_name"), data.get("user_id"), data.get("workspace", str(self.config.workspace))) != (
            self.config.project_name, self.config.user_id, str(self.config.workspace),
        ):
            raise ValueError("检查点不属于当前项目或用户")

    def get_stats(self) -> dict:
        return {
            "project_name": self.config.project_name,
            "session_id": self.session_id,
            "history_messages": len(self.history),
            "trace_events": len(self.trace),
            **self.stats,
        }
