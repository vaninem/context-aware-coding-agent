"""在基础上下文助手上增加项目笔记，不让笔记和会话记忆混为同一对象。"""

from ..config import AppConfig
from ..context import ContextBuilder
from ..ports import ChatModel
from ..tools.memory import MemoryTool
from ..tools.notes import NoteTool
from ..tools.rag import RAGTool
from ..types import ContextPacket, Note
from .base import ContextAwareAgent


class ProjectAssistant(ContextAwareAgent):
    def __init__(
        self,
        config: AppConfig,
        model: ChatModel,
        memory_tool: MemoryTool | None = None,
        rag_tool: RAGTool | None = None,
        context_builder: ContextBuilder | None = None,
        note_tool: NoteTool | None = None,
    ) -> None:
        super().__init__(config, model, memory_tool, rag_tool, context_builder)
        self.note_tool = note_tool if note_tool is not None else NoteTool(config.notes_dir)

    def _notes_to_packets(self, query: str) -> list[ContextPacket]:
        """检索项目笔记，保留来源与最近更新时间。"""
        return [
            ContextPacket(f"{note.title}\n{note.content}", timestamp=note.updated_at, metadata={
                "type": "note", "id": note.id, "note_id": note.id, "note_type": note.note_type,
                "tags": list(note.tags), "updated_at": note.updated_at,
            })
            for note in self.note_tool.search(query, limit=5)
        ]

    def record_note(
        self,
        title: str,
        content: str,
        note_type: str = "general",
        tags: list[str] | None = None,
    ) -> Note:
        """显式写笔记入口，具体存储由 E02 的 NoteTool.create 实现。"""
        return self.note_tool.create(title, content, note_type=note_type, tags=tags)
