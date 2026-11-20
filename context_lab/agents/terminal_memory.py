"""将终端的实际任务结果接入现有记忆服务。"""

import json
from copy import copy
from hashlib import sha256
from pathlib import Path

from ..memory import MemoryService
from ..models import HTTPChatModel
from ..ports import ChatModel
from ..tools.memory import MemoryTool
from ..types import ContextPacket, TraceEvent


class TerminalMemory:
    def __init__(self, root: Path, user_id: str, model: ChatModel, token_counter):
        self.root = root.expanduser().resolve() / sha256(user_id.encode()).hexdigest()
        self.user_id = user_id
        if isinstance(model, HTTPChatModel) and model.json_output:
            model = copy(model)
            model.json_output = False
        self.model = model
        self.token_counter = token_counter

    def project_tool(self, project: Path) -> MemoryTool:
        project_id = sha256(str(project.resolve()).encode()).hexdigest()
        path = self.root / "projects" / project_id / "memories.json"
        return MemoryTool(path, self.user_id, service=MemoryService(
            path, self.user_id, summary_model=self.model, token_counter=self.token_counter,
        ))

    def _preferences(self) -> MemoryService:
        return MemoryService(self.root / "preferences.json", self.user_id)

    def recall(self, service: MemoryService, query: str, budget: int) -> list[ContextPacket]:
        """优先提供最近的交付和未完成项，再补充与提问匹配的旧进度。"""
        recent = sorted(
            (item for item in service.repository.read_snapshot().records if item.memory_type == "project_progress"),
            key=lambda item: (item.created_at, item.id), reverse=True,
        )
        latest = recent[:1]
        if latest and not latest[0].metadata.get("delivered"):
            latest += [item for item in recent if item.metadata.get("delivered")][:1]
        matches = service.search(query, limit=2, memory_type="project_progress", mode="lexical") if recent else []
        preferences = sorted(self._preferences().repository.read_snapshot().records,
                             key=lambda item: (item.created_at, item.id), reverse=True)[:2]
        records = [(item, "项目历史进度", 1.0) for item in latest]
        records.extend((item, "用户明确偏好", 0.9) for item in preferences)
        records.extend((item, "相关项目记录", 0.8) for item in matches)
        seen = set()
        packets = []
        for item, label, relevance in records:
            if item.id in seen or budget <= 0:
                continue
            text = self.token_counter.truncate(
                f"{label}（{item.created_at.date()}）\n{item.content}", min(360, budget),
            )
            if text:
                packets.append(ContextPacket(text, relevance_score=relevance, timestamp=item.created_at,
                                             metadata={"type": "terminal_memory", "id": item.id}))
                budget -= self.token_counter.count(text)
                seen.add(item.id)
        return packets

    def save_turn(self, project: Path, result: dict, session_id: str) -> TraceEvent | None:
        """以回合 ID 去重，只持久化交付结果、待验证任务和明确的用户偏好。"""
        turn_id = result["turn_id"]
        preference = result.get("user_preference", "")
        saved = []
        if preference:
            self._preferences().add(preference, memory_type="preference", importance=0.8,
                                    metadata={"turn_id": turn_id}, memory_id=f"preference_{turn_id}")
            saved.append("用户偏好")
        repair = result.get("repair") or {}
        creation = result.get("project") or {}
        if result["mode"] not in {"repair", "create"}:
            return TraceEvent("memory_write", "已保存终端记忆", {"saved": saved}) if saved else None

        delivered = bool(repair.get("applied_to") or creation.get("status") == "generated")
        status = repair.get("status") or creation.get("status") or "未验证"
        files: list[str] = []
        if delivered:
            files = [item["path"] for item in repair.get("changed_files", [])] or creation.get("files", [])
        verification = {phase: run["status"] for phase, run in repair.get("verification", {}).items()}
        if (run := creation.get("verification")) is not None:
            verification["项目测试"] = run["status"]
        if baseline := result.get("baseline"):
            verification.update({f"修改前/{phase}": run["status"] for phase, run in baseline.items()})
        command = (result.get("test_plan") or {}).get("command") or (result.get("project_settings") or {}).get("command")
        task = self.token_counter.truncate(result["query"], 160)
        lines = [f"交付状态：{'已应用到项目' if delivered else '任务未交付，仍待处理'}（{status}）"]
        lines.append("任务：" + task)
        if files:
            lines.append("已改文件：" + "、".join(files[:12]))
        if verification:
            lines.append("实际验证：" + json.dumps(verification, ensure_ascii=False))
        if not delivered and repair.get("status") in {"failed", "regression_failed", "timeout"}:
            output = self.token_counter.truncate("\n".join(repair["output"].strip().splitlines()[-5:]), 100)
            if output:
                lines.append("最后一次失败日志：" + output)
        self.project_tool(project).service.add(
            "\n".join(lines), memory_type="project_progress", importance=0.8,
            metadata={"turn_id": turn_id, "session_id": session_id, "delivered": delivered,
                      "files": files, "test_command": command, "verification": verification},
            memory_id=f"progress_{turn_id}",
        )
        return TraceEvent("memory_write", "已保存终端记忆", {"saved": [*saved, "项目进度"]})
