"""M01：统一的字典快照与业务对象互相转换，无列表格式迁移分支。"""

from copy import deepcopy
from dataclasses import asdict, fields
from datetime import datetime, timezone
import json
from typing import Any

from .types import MemoryIndexEvent, MemoryRecord, MemorySnapshot, MemorySourceRef


def _copy_fields(value: Any, model: type, location: str) -> dict[str, Any]:
    """校验完整字段再复制，避免 dataclass 用默认值掩盖文件损坏。"""
    expected = {field.name for field in fields(model)}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{location} 字段缺失或包含未知字段")
    return deepcopy(value)


def _restore_time(value: Any, location: str) -> datetime:
    """只恢复带时区的 ISO 时间，不把无时区时间猜成本机时间。"""
    if not isinstance(value, str):
        raise ValueError(f"{location} 必须是带时区的 ISO 时间字符串")
    try:
        # 兼容 Python 3.10 对 ISO UTC 后缀 Z 的处理。
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        result = datetime.fromisoformat(normalized)
        if result.utcoffset() is None:
            raise ValueError("时间缺少时区")
        return result.astimezone(timezone.utc)
    except (ValueError, OverflowError) as error:
        raise ValueError(f"{location} 必须是有效且带时区的 ISO 时间") from error


def _serialize_time(value: Any, location: str) -> str:
    """将带时区 datetime 统一为 UTC ISO 字符串。"""
    if not isinstance(value, datetime):
        raise ValueError(f"{location} 必须是带时区的 datetime")
    try:
        if value.utcoffset() is None:
            raise ValueError("时间缺少时区")
        return value.astimezone(timezone.utc).isoformat()
    except (ValueError, OverflowError) as error:
        raise ValueError(f"{location} 必须是有效且带时区的 datetime") from error


class MemoryCodec:
    schema_version = 1

    @classmethod
    def empty_document(cls) -> dict[str, Any]:
        """已提供：文件不存在时的默认文档，每次返回独立容器。"""
        return {"schema_version": cls.schema_version, "revision": 0, "records": [], "index_events": []}

    def decode(self, payload: Any) -> MemorySnapshot:
        """恢复整个文件，暂不按用户过滤。

        只接受 {schema_version, revision, records, index_events}，不接受列表。
        created_at/updated_at/valid_from/valid_to 的 ISO 字符串转带时区 UTC，
        None 有效期保留。source_refs 恢复为 MemorySourceRef，事件恢复为
        MemoryIndexEvent（其 created_at 也要转换）。恢复记录原有时间，不取当前时间。
        返回全文件 MemorySnapshot，不筛用户、不写文件、不生成向量；坏数据明确报错。
        """
        expected = {"schema_version", "revision", "records", "index_events"}
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("记忆文件必须包含 schema_version/revision/records/index_events")
        if type(payload["schema_version"]) is not int or payload["schema_version"] != self.schema_version:
            raise ValueError("不支持的记忆文件 schema_version")
        revision = payload["revision"]
        if type(revision) is not int or revision < 0:
            raise ValueError("revision 必须是非负整数")
        if not isinstance(payload["records"], list) or not isinstance(payload["index_events"], list):
            raise ValueError("records 和 index_events 必须是列表")

        records = []
        record_ids = set()
        for position, record_raw in enumerate(payload["records"]):
            location = f"records[{position}]"
            data = _copy_fields(record_raw, MemoryRecord, location)
            for key in ("id", "content", "user_id", "memory_type"):
                if not isinstance(data[key], str) or not data[key].strip():
                    raise ValueError(f"{location}.{key} 必须是非空字符串")
            for key in ("task_id", "fact_key", "supersedes_id"):
                if data[key] is not None and (not isinstance(data[key], str) or not data[key].strip()):
                    raise ValueError(f"{location}.{key} 必须为 None 或非空字符串")
            if type(data["version"]) is not int or data["version"] < 1:
                raise ValueError(f"{location}.version 必须是正整数")
            importance = data["importance"]
            if type(importance) not in (int, float) or not 0 <= importance <= 1:
                raise ValueError(f"{location}.importance 必须是 0 到 1 之间的有限数值")
            if data["layer"] not in ("working", "summary", "archive"):
                raise ValueError(f"{location}.layer 不受支持")
            if data["layer"] == "working" and data["task_id"] is None:
                raise ValueError(f"{location} 的 working 记忆必须有 task_id")
            if data["summary_state"] not in ("fresh", "stale"):
                raise ValueError(f"{location}.summary_state 不受支持")
            if not isinstance(data["metadata"], dict):
                raise ValueError(f"{location}.metadata 必须是 JSON 字典")
            try:
                restored = json.loads(json.dumps(data["metadata"], allow_nan=False))
                if restored != data["metadata"]:
                    raise ValueError("元数据不能无损保存为 JSON")
            except (TypeError, ValueError, OverflowError, RecursionError) as error:
                raise ValueError(f"{location}.metadata 必须能无损保存为标准 JSON") from error

            for key in ["created_at", "updated_at", "valid_from", "valid_to"]:
                if data[key] is None and key in ("valid_from", "valid_to"):
                    continue
                data[key] = _restore_time(data[key], f"{location}.{key}")
            if data["updated_at"] < data["created_at"]:
                raise ValueError(f"{location}.updated_at 不能早于 created_at")
            start, end = data["valid_from"], data["valid_to"]
            if start is not None and end is not None and start >= end:
                raise ValueError(f"{location} 的 valid_from 必须早于 valid_to")

            if not isinstance(data["source_refs"], list):
                raise ValueError(f"{location}.source_refs 必须是列表")
            memory_refs = []
            for ref_position, raw_ref in enumerate(data["source_refs"]):
                ref_location = f"{location}.source_refs[{ref_position}]"
                ref = _copy_fields(raw_ref, MemorySourceRef, ref_location)
                if not isinstance(ref["memory_id"], str) or not ref["memory_id"].strip():
                    raise ValueError(f"{ref_location}.memory_id 必须是非空字符串")
                if type(ref["version"]) is not int or ref["version"] < 1:
                    raise ValueError(f"{ref_location}.version 必须是正整数")
                memory_refs.append(MemorySourceRef(**ref))
            data["source_refs"] = memory_refs

            identity = (data["user_id"], data["id"])
            if identity in record_ids:
                raise ValueError(f"{location} 与同用户的记忆 ID 重复")
            record_ids.add(identity)
            records.append(MemoryRecord(**data))

        events = []
        event_ids = set()
        for position, event_raw in enumerate(payload["index_events"]):
            location = f"index_events[{position}]"
            data = _copy_fields(event_raw, MemoryIndexEvent, location)
            for key in ("id", "user_id", "memory_id"):
                if not isinstance(data[key], str) or not data[key].strip():
                    raise ValueError(f"{location}.{key} 必须是非空字符串")
            if data["operation"] not in ("upsert", "delete"):
                raise ValueError(f"{location}.operation 必须为 upsert 或 delete")
            for key, minimum in (("version", 1), ("attempts", 0)):
                if type(data[key]) is not int or data[key] < minimum:
                    raise ValueError(f"{location}.{key} 必须是大于等于 {minimum} 的整数")
            if data["error_code"] is not None and (
                not isinstance(data["error_code"], str) or not data["error_code"].strip()
            ):
                raise ValueError(f"{location}.error_code 必须为 None 或非空字符串")
            data["created_at"] = _restore_time(data["created_at"], f"{location}.created_at")
            identity = (data["user_id"], data["id"])
            if identity in event_ids:
                raise ValueError(f"{location} 与同用户的事件 ID 重复")
            event_ids.add(identity)
            events.append(MemoryIndexEvent(**data))
        return MemorySnapshot(revision=revision, records=records, index_events=events)

    def encode(self, snapshot: MemorySnapshot) -> dict[str, Any]:
        """按唯一 envelope 格式序列化；不修改传入对象。

        创建时间使用 created_at，修改时间使用 updated_at，不写 timestamp 字段。
        时间存 ISO 字符串，None 保留，引用和事件转普通字典，记录 revision。
        结果必须可被标准 JSON 无损读回，不序列化模型、索引或连接对象。
        """
        if not isinstance(snapshot, MemorySnapshot):
            raise ValueError("snapshot 必须是 MemorySnapshot")
        if not isinstance(snapshot.records, list) or not isinstance(snapshot.index_events, list):
            raise ValueError("records 和 index_events 必须是列表")
        for position, record in enumerate(snapshot.records):
            if not isinstance(record, MemoryRecord):
                raise ValueError(f"records[{position}] 必须是 MemoryRecord")
            if not isinstance(record.source_refs, list) or any(
                not isinstance(ref, MemorySourceRef) for ref in record.source_refs
            ):
                raise ValueError(f"records[{position}].source_refs 必须是 MemorySourceRef 列表")
        for position, event in enumerate(snapshot.index_events):
            if not isinstance(event, MemoryIndexEvent):
                raise ValueError(f"index_events[{position}] 必须是 MemoryIndexEvent")
        try:
            data = asdict(snapshot)
        except (TypeError, ValueError, RecursionError) as error:
            raise ValueError("记忆快照包含无法复制或序列化的数据") from error

        for position, record in enumerate(data["records"]):
            for key in ["created_at", "updated_at", "valid_from", "valid_to"]:
                if record[key] is None and key in ("valid_from", "valid_to"):
                    continue
                record[key] = _serialize_time(record[key], f"records[{position}].{key}")
        for position, event in enumerate(data["index_events"]):
            event["created_at"] = _serialize_time(event["created_at"], f"index_events[{position}].created_at")
        data["schema_version"] = self.schema_version
        # 复用已有字段和业务数据校验，保证返回的格式能被 decode 正确恢复。
        self.decode(data)
        return data
