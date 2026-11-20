"""自然语言任务分流及生成项目的模型协议。"""

import json
from pathlib import Path
from time import sleep

from ..models import ModelRequestError
from ..tools.project_files import project_path
from ..types import Message

ROUTER_PROMPT = """你负责理解当前终端用户的意图，并选择本轮动作。
analyze 用于问答、检查、解释、给方案或仅分析，包含「怎么做」等未授权写入的提问。
edit 用于用户请求实现、修复或改动已有项目。优先使用明确指定的目录，否则使用当前项目。
create 用于用户要求从零新建独立项目。根目录是空目录且用户没指定名字时可用 "."。
根目录有内容时，请为新项目选一个合适的新的相对子目录。新项目的后续改动使用 edit。
历史对话仅供理解「刚才」「继续」「按这个做」等指代。当前请求优先，遇到确实无法判断的目标才用 clarify。
只返回 JSON 对象 {"action":"analyze|edit|create|clarify","target":"相对工作区路径","message":"必要时给用户的一个澄清问题","require_new_tests":false,"user_preference":""}。
target 可以是 "." 或工作区内子目录。message 只在 clarify 时使用，其他动作填空串。
对于 edit，用户明确要求新增、补写测试用例时 require_new_tests 为 true，例如“加删除功能，顺便补测试”。
只要求运行已有测试、检查结果时为 false。其他动作填 false。
用户明确表达长期个人偏好时，将本轮请求里对应的原文片段逐字填入 user_preference，例如“我以后写 Python 测试优先用 pytest”。
询问已有偏好、要求执行当前任务或只表达项目局部要求时，user_preference 填空串。"""

BUILD_PROMPT = """你负责把用户需求实现成一个完整、可运行的新项目。
工具在隔离目录累计本轮生成的文件；验证通过后才会交付。请自行决定文件结构。
包含 README 和可运行的测试，测试应检验用户要求的实际行为。测试环境无网络，依赖必须已在镜像中。
输出单个 JSON 对象 {"files":{"相对文件路径":"完整的 UTF-8 正文"},"remove":[],"verify":true}，所有换行以 JSON 字符串转义。
文件路径从新项目根目录开始，例如 README.md、src/app.py、tests/test_app.py。新项目的目录名由程序处理，路径前面无需再加项目名。
files 只列本次新增或改写的文件，未列文件将保留；remove 只用于移除本轮草稿中的文件。
大项目分多次提交，未完成时 verify 为 false；代码和测试就绪后设为 true。
测试失败时根据完整任务和反馈修正草稿。先修复代码，测试需要覆盖需求。"""


def read_model_object(text: str) -> dict:
    body = text.strip()
    if body.startswith("```json\n") and body.endswith("```"):
        body = body[8:-3].strip()
    payload = json.loads(body)
    if not isinstance(payload, dict):
        raise ValueError("需要返回一个 JSON 对象")
    return payload


def read_project_update(text: str) -> tuple[dict, list, bool]:
    """读取首个文件提交对象，兼容模型附带的说明和文件数组。"""
    body = text.strip()
    if body.startswith("```json\n"):
        body = body[8:].lstrip()
    payload, _ = json.JSONDecoder().raw_decode(body)
    if not isinstance(payload, dict) or "files" not in payload:
        raise ValueError("项目提交需要包含 files")
    files, removed, verify = payload["files"], payload.get("remove", []), payload.get("verify", True)
    if isinstance(files, list):
        written = {}
        for item in files:
            if (not isinstance(item, dict) or not isinstance(item.get("path"), str)
                    or "content" not in item or item["path"] in written):
                raise ValueError("文件数组的每项需要不同的 path 和一个 content")
            written[item["path"]] = item["content"]
        files = written
    if type(verify) is not bool:
        raise ValueError("verify 必须为布尔值")
    return files, removed, verify


def generate_with_retry(model, messages: list[Message]) -> str:
    for attempt in range(3):
        try:
            return model.generate(messages)
        except ModelRequestError as exc:
            if not exc.retryable or attempt == 2:
                raise
            sleep(attempt + 1)


def route_request(agent, query: str) -> tuple[str, str, str, bool, str]:
    """以当前目录和最近对话为依据，让模型选择动作与目标。"""
    root = agent.config.workspace
    active = agent.active_project
    try:
        current = project_path(root, active)
    except ValueError:
        current = root
    if not current.is_dir():
        current = root

    def listing(directory: Path) -> list[str]:
        result = []
        for path in sorted(directory.iterdir()):
            if path.name.startswith(".") or path.is_symlink() or path == agent.config.state_dir:
                continue
            result.append(path.name + ("/" if path.is_dir() else ""))
            if len(result) == 60:
                break
        return result

    history = agent.history
    if agent.terminal_memory is not None:
        history = history[agent.history_start:] if active == agent.history_project else []
    request = {
        "workspace": str(root), "workspace_empty": not any(root.iterdir()),
        "current_project": current.relative_to(root).as_posix(),
        "workspace_entries": listing(root), "current_project_entries": listing(current),
        "recent_history": [{"role": item.role, "text": item.content[:1800]} for item in history[-6:]],
        "request": query,
    }
    answer = generate_with_retry(agent.model, [Message("system", ROUTER_PROMPT),
                                               Message("user", json.dumps(request, ensure_ascii=False))])
    try:
        payload = read_model_object(answer)
        require_new_tests = payload.pop("require_new_tests", False)
        preference = payload.pop("user_preference", "")
        if not isinstance(preference, str) or not preference.strip() or preference not in query or len(preference) > 512:
            preference = ""
        if (type(require_new_tests) is not bool or set(payload) != {"action", "target", "message"}
                or not all(isinstance(item, str) for item in payload.values())):
            raise ValueError("路由格式无效")
        action, target, message = (payload[key] for key in ("action", "target", "message"))
        if action == "clarify" and message.strip():
            return action, current.relative_to(root).as_posix(), message, False, preference
        if action not in {"analyze", "edit", "create"}:
            raise ValueError("路由动作无效")
        target_path = project_path(root, target)
        state_dir = agent.config.state_dir
        if target_path.is_relative_to(state_dir):
            raise ValueError("该目录用于保存会话状态")
        if action == "create":
            if target_path.exists() and (not target_path.is_dir() or any(target_path.iterdir())):
                return "clarify", current.relative_to(root).as_posix(), "这个目标已有内容。新项目要放到哪个子目录？", False, preference
        elif not target_path.is_dir():
            return "clarify", current.relative_to(root).as_posix(), "目标目录暂时找不到。你想修改哪个已有项目，还是在这个目录下新建？", False, preference
        return action, target_path.relative_to(root).as_posix(), "", action == "edit" and require_new_tests, preference
    except (AttributeError, TypeError, ValueError):
        return "clarify", current.relative_to(root).as_posix(), "请告诉我这次要生成独立项目，还是改动某个已有目录。", False, ""
