"""同一仓库会话的终端入口和进度展示。"""

import json
import shlex
from collections.abc import Callable
from dataclasses import asdict

from .agents.codebase_maintainer import CodebaseMaintainer
from .storage.json_store import JsonFileStore
from .types import AgentReply, TraceEvent


COMMANDS = "直接描述你想分析、修改或新建的项目。/status 查看上下文与状态，/resume 恢复，/exit 退出。"
PHASES = {"prepare_context": "准备上下文", "check_baseline": "复现", "decide": "模型决策",
          "execute_tool": "执行工具", "finish_repair": "整理修复结果", "route_request": "识别任务",
          "freeze_repair": "冻结现有项目", "discover_tests": "发现测试", "draft_project": "生成文件", "verify_project": "验证新项目",
          "finish_project": "交付新项目", "render_answer": "生成最终答复"}


def display_progress(event: TraceEvent) -> None:
    metadata = event.metadata
    details = [str(metadata[key]) for key in ("tool", "location", "mode", "status") if key in metadata]
    if "submission" in metadata:
        details.append(f"第 {metadata['submission']} 次提交")
    if event.stage == "baseline":
        details.extend(f"{name}={status}" for name, status in metadata.items())
    if event.stage == "test_plan" and metadata.get("command"):
        details += [shlex.join(metadata["command"]), f"依据 {metadata['source']}"]
    if event.stage == "context_compaction":
        details.append(f"本次覆盖 {metadata['newly_covered']} 条消息")
    if verification := metadata.get("verification"):
        details.extend(f"{name}={status}" for name, status in verification.items())
    if (submission_error := metadata.get("submission_error")) and (reason := submission_error.get("message")):
        details.append(reason.partition("\n")[0][:220])
    for hit in metadata.get("hits", [])[:3]:
        details.append(f"{hit['source']}:{hit['line']}")
    message = " · ".join([event.message, *details])
    print(f"[{event.timestamp.astimezone():%H:%M:%S}] {message}", flush=True)


def repair_report(reply: AgentReply) -> dict:
    return {
        "baseline": reply.baseline,
        "result": reply.repair,
        "retrieval": [{"stage": event.stage, **event.metadata}
                      for event in reply.trace if event.stage in {"memory_index", "rag_index"}],
        "trace": [{**asdict(event), "timestamp": event.timestamp.isoformat()} for event in reply.trace],
    }


def _display_status(agent: CodebaseMaintainer) -> None:
    status = agent.session_status()
    phase = "、".join(PHASES.get(node, node) for node in status["next"])
    if not phase:
        phase = "整理上次结果" if status["pending"] else "空闲"
    print(f"已完成 {status['turns']} 轮。当前阶段 {phase}。")
    print(f"当前项目 {status['workspace']}")
    if status.get("target") or status.get("request"):
        print(f"本轮任务 {status.get('request', '')} · {status.get('target') or '.'}")
    if validation := status.get("validation"):
        print(f"最近验证为第 {validation.get('submission', '?')} 次提交，结果 {validation['status']}。")
    elif status.get("result"):
        print(f"最近修复结果 {status['result']}。")
    if status.get("project"):
        print(f"项目生成结果 {status['project']}。")
    if (plan := status.get("test_plan")) and plan["command"]:
        print(f"测试命令 {shlex.join(plan['command'])} · 依据 {plan['source']}")
    if status["pending"] and status.get("draft_files"):
        print("已保存草稿文件 " + "、".join(status["draft_files"][:8]))
    if context := status.get("context"):
        if budget := status.get("context_budget"):
            print(f"本轮上下文约 {context['tokens']}/{budget['input_budget']} token，摘要触发线 {budget['trigger_tokens']} token，"
                  f"已摘要 {budget['covered_messages']} 条消息。")
            if budget.get("fallback"):
                print("本轮已使用预算内的最近对话。")
            print(context["preview"])
        else:
            print(f"本轮上下文约 {context['tokens']} token，以下为预览。\n{context['preview']}")
    if status["history"]:
        print("最近对话")
        for message in status["history"]:
            print(f"  {message['role']} · {message['text']}")


def _display_reply(agent: CodebaseMaintainer, reply: AgentReply, *, show_answer: bool = True) -> None:
    if show_answer:
        print(reply.answer)
    if agent.completed_turn_id is None or (reply.repair is None and reply.project is None):
        return
    directory = agent.config.state_dir / "artifacts" / agent.completed_turn_id
    path = directory / "report.json"
    if reply.project is not None:
        report = repair_report(reply)
        report["project"] = reply.project
        JsonFileStore(path).write(report)
        print(f"生成报告 {path}")
        if reply.project["status"] == "generated":
            print(f"项目目录 {reply.project['target']}")
        return
    JsonFileStore(path).write(repair_report(reply))
    print(f"修复报告 {path}")
    if reply.repair is None:
        return
    result = reply.repair
    print(f"修复状态 {result['status']}")
    if result.get("applied_to"):
        print(f"修改已写入 {result['applied_to']}")
    for change in result.get("changed_files", []):
        print(f"  {change['path']} +{change['added']} -{change['removed']}")
    if result["status"] == "passed" and result["patch"]:
        patch_path = directory / "repair.patch"
        patch_path.write_text(result["patch"], encoding="utf-8")
        print(f"补丁 {patch_path}")


def _run_and_display(agent: CodebaseMaintainer, turn: Callable[[], AgentReply], *, stream: bool) -> None:
    last_character = ""

    def write_chunk(chunk: str | None) -> None:
        nonlocal last_character
        if chunk is None:
            if last_character and last_character != "\n":
                print(flush=True)
            return
        last_character = chunk[-1]
        print(chunk, end="", flush=True)

    previous = agent.answer_output
    agent.answer_output = write_chunk if stream else None
    try:
        reply = turn()
    finally:
        agent.answer_output = previous
    _display_reply(agent, reply, show_answer=not last_character)


def run_terminal(agent: CodebaseMaintainer, *, new_session: bool = False, stream: bool = True) -> int:
    """载入会话，系统命令在本地处理，其余输入交给模型分流。"""
    if not agent.config.workspace.is_dir():
        raise ValueError("目标仓库目录不存在")
    if agent.config.checkpoint_path.is_file():
        agent.load_checkpoint()
        if new_session:
            agent.start_new_session()
    print(f"工作区 {agent.config.workspace}")
    print(COMMANDS)
    if agent.session_status()["pending"]:
        print("有未完成的回合。输入 /resume 从已保存的进度继续。")
    while True:
        try:
            line = input("agent> ").strip()
        except EOFError:
            return 0
        except KeyboardInterrupt:
            print()
            continue
        if not line:
            continue
        try:
            if line == "/exit":
                return 0
            if line == "/help":
                print(COMMANDS)
            elif line == "/status":
                _display_status(agent)
            elif line == "/resume":
                status = agent.session_status()
                if not status["pending"]:
                    print("当前会话没有待恢复的回合。")
                    continue
                _run_and_display(agent, agent.resume, stream=stream)
            elif line.startswith("/"):
                print("未知命令。输入 /help 查看用法。")
            else:
                _run_and_display(agent, lambda: agent.converse(line), stream=stream)
        except KeyboardInterrupt:
            print("\n当前回合已中断。输入 /status 查看检查点，或 /resume 继续。")
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"本轮结束 · {exc}")
