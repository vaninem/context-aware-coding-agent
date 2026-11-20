"""Memory 工具入口：统一委托 MemoryService，业务练习为 M01—M08。"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..memory import MemoryService
from ..memory.types import MemoryMutation, MemoryRecord
from ..types import HistorySummaryRequest, Message, ToolResult
from .base import BaseTool


class MemoryTool(BaseTool):
    def __init__(self, store_path: Path, user_id: str = "default", *, service: MemoryService | None = None):
        """装配唯一 MemoryService；不读取文件、不生成记忆或连接模型。"""
        super().__init__("memory", "分层与时间记忆：业务管理、检索和摘要回查")
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id 必须为非空字符串")
        self.store_path = Path(store_path)
        self.user_id = user_id
        self.service = service if service is not None else MemoryService(self.store_path, user_id)
        if self.service.user_id != user_id or self.service.path.resolve() != self.store_path.resolve():
            raise ValueError("MemoryService 必须绑定相同用户和业务文件")

    def run(self, parameters: dict[str, Any]) -> ToolResult:
        """工具参数解析与分发已提供，业务行为由服务实现；调用者不能替换用户。"""
        if not isinstance(parameters, dict):
            raise ValueError("工具参数必须为字典")
        routes = {
            "add": (self.add, {"content", "memory_type", "importance", "metadata", "layer", "task_id", "valid_from", "valid_to", "fact_key"}, {"content"}),
            "search": (self.search, {"query", "limit", "memory_type", "min_importance", "max_importance", "mode", "as_of", "layers", "task_id"}, {"query"}),
            "update": (self.update, {"memory_id", "content", "importance", "metadata"}, {"memory_id"}),
            "remove": (self.remove, {"memory_id"}, {"memory_id"}),
            "revise_fact": (self.service.revise_fact, {"memory_id", "content", "effective_from"}, {"memory_id", "content", "effective_from"}),
            "finish_task": (self.service.finish_task, {"task_id"}, {"task_id"}),
            "expand_summary": (self.service.expand_summary, {"summary_id"}, {"summary_id"}),
            "rebuild_summary": (self.service.rebuild_summary, {"summary_id"}, {"summary_id"}),
            "prepare_context": (self.service.prepare_context, {"query", "task_id", "max_tokens", "as_of"}, {"query", "task_id"}),
            "plan_history_cleanup": (self.service.plan_history_cleanup, {"limit"}, set()),
            "cleanup_history": (self.service.cleanup_history, {"dry_run", "limit"}, set()),
        }
        action = parameters.get("action")
        if not isinstance(action, str) or action not in routes:
            raise ValueError("不支持的 memory action")
        function, allowed, required = routes[action]
        arguments = {key: value for key, value in parameters.items() if key != "action"}
        if set(arguments) - allowed or required - set(arguments):
            raise ValueError("memory 参数字段不符合所选 action")
        for name in ("as_of", "valid_from", "valid_to", "effective_from"):
            if name in arguments and isinstance(arguments[name], str):
                value = datetime.fromisoformat(arguments[name])
                if value.utcoffset() is None:
                    raise ValueError("记忆时间必须带时区")
                arguments[name] = value.astimezone(timezone.utc)
        if isinstance(arguments.get("layers"), list):
            arguments["layers"] = tuple(arguments["layers"])
        data = function(**arguments)
        return ToolResult(success=True, data=data, metadata={"tool": self.name, "action": action})

    def add(
        self, content: str, memory_type: str = "episodic", importance: float = 0.5,
        metadata: dict[str, Any] | None = None, **options: Any,
    ) -> MemoryRecord:
        return self.service.add(content, memory_type, importance, metadata, **options)

    def search(self, query: str, limit: int = 5, **filters: Any) -> list[MemoryRecord]:
        return self.service.search(query, limit, **filters)

    def update(
        self, memory_id: str, content: str | None = None, importance: float | None = None,
        *, metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        return self.service.update(memory_id, content, importance, metadata=metadata)

    def remove(self, memory_id: str) -> bool:
        return self.service.remove(memory_id)

    def checkpoint_task(self, request: HistorySummaryRequest) -> MemoryRecord:
        """供 Agent 内部调用的强类型入口；不把原始 Message 对象作为 JSON 工具参数。"""
        return self.service.checkpoint_task(request)

    def propose(self, messages: list[Message]) -> list[MemoryMutation]:
        return self.service.propose(messages)

    def apply_mutations(self, mutations: list[MemoryMutation]) -> list[MemoryRecord]:
        return self.service.apply_mutations(mutations)
