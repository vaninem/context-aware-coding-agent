"""E02：Markdown 正文、YAML 前置元数据与本地索引的笔记练习。"""

import json
import os
import re
import tempfile
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..errors import ExercisePending
from ..storage.json_store import JsonFileStore
from ..types import Note, ToolResult, utc_now
from .base import BaseTool


class NoteTool(BaseTool):
    """项目笔记骨架；summary 只统计数量与类型，不调用语言模型。"""

    def __init__(self, workspace: Path):
        """保存笔记目录和索引位置；目录创建/索引恢复留给 E02 按需完成。"""
        super().__init__("notes", "Markdown 项目笔记及类型/标签检索")
        self.workspace = Path(workspace)
        self.index_path = self.workspace / "notes_index.json"
        self.index: dict[str, dict[str, Any]] = {}
        self.store = JsonFileStore(self.index_path)
        self._loaded = False

    def run(self, parameters: dict[str, Any]) -> ToolResult:
        """分发 create/read/update/search/list_notes/delete/summary，不吞练习异常。"""
        if not isinstance(parameters, dict):
            raise ValueError("工具参数必须为字典")
        action = parameters.get("action")
        routes = {
            "create": (self.create, {"title", "content", "note_type", "tags"}, {"title", "content"}),
            "read": (self.read, {"note_id"}, {"note_id"}),
            "update": (self.update, {"note_id", "title", "content", "note_type", "tags"}, {"note_id"}),
            "search": (self.search, {"query", "limit", "note_type", "tags"}, {"query"}),
            "list_notes": (self.list_notes, {"limit", "note_type"}, set()),
            "delete": (self.delete, {"note_id"}, {"note_id"}),
            "summary": (self.summary, set(), set()),
        }
        if not isinstance(action, str) or action not in routes:
            raise ValueError("notes action 不受支持")
        function, allowed, required = routes[action]
        arguments = {key: value for key, value in parameters.items() if key != "action"}
        if set(arguments) - allowed or required - set(arguments):
            raise ValueError("notes 参数字段不符合所选 action")
        data = function(**arguments)
        return ToolResult(success=True, data=data, metadata={"tool": self.name, "action": action})

    def create(
        self, title: str, content: str, note_type: str = "general", tags: list[str] | None = None
    ) -> Note:
        """保存笔记和索引；前置元数据仅支持逐行 JSON 值组成的 YAML 子集。"""
        for label, value in (("标题", title), ("正文", content), ("笔记类型", note_type)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label}必须是非空字符串")
        if tags is not None:
            if not isinstance(tags, list):
                raise ValueError("tags 必须是字符串列表")
            if any(not isinstance(tag, str) or not tag.strip() for tag in tags):
                raise ValueError("每个标签必须是非空字符串")

        # 在创建文件之前读取索引，已有索引损坏时直接报错。
        index = self.store.read(default={})
        if not isinstance(index, dict):
            raise ValueError("笔记索引必须是字典")

        note_id = f"note_{uuid4().hex}"
        file_name = f"{note_id}.md"
        now = utc_now()
        file_path = self.workspace / file_name
        note = Note(
            id=note_id,
            title=title,
            content=content,
            note_type=note_type,
            tags=list(tags) if tags is not None else [],
            created_at=now,
            updated_at=now,
        )
        metadata = {
            "id": note.id,
            "title": note.title,
            "note_type": note.note_type,
            "tags": note.tags,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
        }
        header = "\n".join(
            f"{key}: {json.dumps(value, ensure_ascii=False)}"
            for key, value in metadata.items()
        )
        index[note_id] = {**metadata, "file_path": file_name}

        self.workspace.mkdir(parents=True, exist_ok=True)
        # x 模式仅创建新文件，避免意外覆盖已有笔记。
        destination = file_path.open("x", encoding="utf-8")
        try:
            with destination:
                destination.write(f"---\n{header}\n---\n\n{note.content}")
            self.store.write(index)
        except Exception:
            # 尽量清理新文件；清理失败也保留原始写入异常。
            with suppress(OSError):
                file_path.unlink(missing_ok=True)
            raise

        self.index = index
        self._loaded = True
        return note

    def read(self, note_id: str) -> Note:
        """校验 ID、路径、元数据与索引，保留正文并将时间恢复为 UTC。"""
        if not isinstance(note_id, str) or not re.fullmatch(r"note_[0-9a-f]{32}", note_id):
            raise ValueError("笔记 ID 格式错误")

        index = self.store.read(default={})
        if not isinstance(index, dict):
            raise ValueError("笔记索引必须是字典")

        entry = index.get(note_id)
        if entry is None:
            raise ValueError(f"找不到笔记 {note_id}")
        if not isinstance(entry, dict):
            raise ValueError("笔记索引项格式错误")
        fields = {"id", "title", "note_type", "tags", "created_at", "updated_at"}
        if not fields.issubset(entry):
            raise ValueError("笔记索引项缺少必要字段")
        file_name = entry.get("file_path")
        if not isinstance(file_name, str) or file_name != f"{note_id}.md":
            raise ValueError("笔记文件名与 ID 不一致")

        workspace = self.workspace.resolve()
        file_path = (workspace / file_name).resolve()

        # 防止路径穿越或符号链接指向工作区外
        if file_path.parent != workspace:
            raise ValueError("笔记路径超出工作区")

        # 禁止自动转换换行符，确保正文的 CRLF 等原始字符也能保留。
        with file_path.open("r", encoding="utf-8", newline="") as source:
            text = source.read()
        if not text.startswith("---\n"):
            raise ValueError("笔记缺少元数据起始标记")

        metadata_text, boundary, content = text[len("---\n"):].partition("\n---\n\n")
        if not boundary:
            raise ValueError("笔记缺少元数据结束标记")
        metadata = {}
        for line in metadata_text.splitlines():
            key, separator, value = line.partition(": ")
            if not separator or not key or key in metadata:
                raise ValueError("笔记元数据格式错误或字段重复")
            try:
                metadata[key] = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError(f"笔记元数据 {key} 不是合法 JSON 值") from exc

        if metadata.keys() != fields:
            raise ValueError("笔记元数据字段缺失或包含不支持的字段")
        if metadata["id"] != note_id or entry["id"] != note_id:
            raise ValueError("笔记 ID 不一致")
        for key in ("title", "note_type"):
            if not isinstance(metadata[key], str) or not metadata[key].strip():
                raise ValueError(f"笔记 {key} 必须是非空字符串")
        tags = metadata["tags"]
        if not isinstance(tags, list) or any(not isinstance(tag, str) or not tag.strip() for tag in tags):
            raise ValueError("笔记 tags 必须是非空字符串组成的列表")
        if not content.strip():
            raise ValueError("笔记正文不能为空")
        if any(entry[key] != metadata[key] for key in fields):
            raise ValueError("笔记元数据与索引不一致")

        for key in ("created_at", "updated_at"):
            if not isinstance(metadata[key], str):
                raise ValueError(f"{key} 必须是 ISO 时间字符串")
            try:
                dt = datetime.fromisoformat(metadata[key])
            except ValueError as exc:
                raise ValueError(f"{key} 不是合法 ISO 时间") from exc
            if dt.utcoffset() is None:
                raise ValueError(f"{key} 必须包含时区")
            metadata[key] = dt.astimezone(timezone.utc)
        if metadata["updated_at"] < metadata["created_at"]:
            raise ValueError("更新时间不能早于创建时间")

        self.index = index
        self._loaded = True
        return Note(content=content, **metadata)

    def update(
        self,
        note_id: str,
        title: str | None = None,
        content: str | None = None,
        note_type: str | None = None,
        tags: list[str] | None = None,
    ) -> Note:
        """更新非 None 字段；先替换笔记，再保存索引，索引写失败时恢复原文。"""
        changes = {
            "title": title,
            "content": content,
            "note_type": note_type,
            "tags": tags,
        }
        # 先校验所有更新值，避免把非法数据写进笔记或索引。
        for key, value in changes.items():
            if value is None:
                continue
            if key == "tags":
                if not isinstance(value, list) or any(
                    not isinstance(tag, str) or not tag.strip() for tag in value
                ):
                    raise ValueError("tags 必须是非空字符串组成的列表")
            elif not isinstance(value, str) or not value.strip():
                raise ValueError(f"{key} 必须是非空字符串")

        note = self.read(note_id)
        for key, value in changes.items():
            if value is not None:
                setattr(note, key, list(value) if key == "tags" else value)

        # 即使系统时间回拨，也不让更新时间早于原来的时间。
        note.updated_at = max(utc_now(), note.updated_at)
        metadata = {
            "id": note.id,
            "title": note.title,
            "note_type": note.note_type,
            "tags": list(note.tags),
            "created_at": note.created_at.isoformat(),
            "updated_at": note.updated_at.isoformat(),
        }
        # read() 已加载并检查索引；复制索引项，保存成功前不修改内存索引。
        index = {**self.index, note_id: {**self.index[note_id], **metadata}}
        header = "\n".join(
            f"{key}: {json.dumps(value, ensure_ascii=False)}"
            for key, value in metadata.items()
        )
        markdown_text = f"---\n{header}\n---\n\n{note.content}"
        workspace = self.workspace.resolve()
        file_path = (workspace / index[note_id]["file_path"]).resolve()
        if file_path.parent != workspace:
            raise ValueError("笔记路径超出工作区")
        old_data = file_path.read_bytes()
        temporary_path = None
        replaced = False

        try:
            # 完整写入同目录临时文件后才替换，写失败不会清空原笔记。
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=file_path.parent, delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(markdown_text.encode("utf-8"))
                temporary.flush()
                os.fsync(temporary.fileno())
            temporary_path.replace(file_path)
            replaced = True
            self.store.write(index)
        except Exception:
            if replaced:
                try:
                    # 原临时路径已被移走，重新写入旧内容后替换回去。
                    with temporary_path.open("wb") as backup:
                        backup.write(old_data)
                        backup.flush()
                        os.fsync(backup.fileno())
                    temporary_path.replace(file_path)
                except OSError as exc:
                    raise RuntimeError(f"更新失败，且无法恢复原笔记：{file_path}") from exc
            raise
        finally:
            # 只清理临时文件；清理失败不覆盖原始保存/恢复异常。
            with suppress(OSError):
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)

        self.index = index
        return note

    def search(
        self,
        query: str,
        limit: int = 5,
        note_type: str | None = None,
        tags: list[str] | None = None,
    ) -> list[Note]:
        """按标题/正文词法匹配查找笔记，并限定类型与标签。

        TODO 1：验证参数，明确多标签采用全部匹配还是任一匹配，并在测试中固定。
        TODO 2：按需恢复索引，先过滤类型/标签，再对标题与正文计算可解释匹配分数。
        TODO 3：稳定排序、截 limit、恢复为 Note；无匹配才返回空列表。
        """
        raise ExercisePending("E02", "实现类型/标签过滤与笔记词法搜索")

    def list_notes(self, limit: int = 20, note_type: str | None = None) -> list[Note]:
        """从磁盘恢复笔记，按类型过滤、更新时间及 ID 降序排列后截取。"""
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError("limit 必须是非负整数")
        if note_type is not None and (
            not isinstance(note_type, str) or not note_type.strip()
        ):
            raise ValueError("note_type 必须是非空字符串或 None")

        index = self.store.read(default={})
        if not isinstance(index, dict):
            raise ValueError("笔记索引必须是字典")
        notes = []
        for note_id in index:
            note = self.read(note_id)
            if note_type is None or note_type == note.note_type:
                notes.append(note)
        notes.sort(
            key=lambda note: (note.updated_at, note.id),
            reverse=True,
        )
        self.index = index
        self._loaded = True
        return notes[:limit]

    def delete(self, note_id: str) -> bool:
        """删除笔记文件及索引项；已不存在时返回 False。

        TODO 1：验证 ID 和工作区边界，读取索引，确定目标。
        TODO 2：删除对应文件并同步索引；定义部分失败时的可诊断行为。
        TODO 3：重复删除返回 False，不能借此吞掉权限或写入异常。
        """
        raise ExercisePending("E02", "实现笔记删除、索引同步与幂等语义")

    def summary(self) -> dict[str, Any]:
        """返回数量与笔记类型统计，不生成 LLM 摘要。

        TODO 1：按需恢复索引，统计总数与每种 note_type 的数量。
        TODO 2：固定返回 total_notes（总数）、type_distribution（类型计数映射）、
        recent_notes（按更新时间排序的近期笔记信息）；空库的零统计是完成后的真实结果。
        TODO 3：统计应与 create/update/delete 后的 list_notes 一致。
        """
        raise ExercisePending("E02", "实现笔记数量与类型统计")
