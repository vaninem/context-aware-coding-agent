"""记忆记录、检索、摘要和索引同步的业务入口。"""

import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from uuid import NAMESPACE_URL, uuid4, uuid5

from ..errors import ExercisePending
from ..ports import ChatModel
from ..types import HistorySummaryRequest, Message, history_prefix_hash, utc_now
from .config import MemoryConfig
from .errors import MemoryConflict, MemoryEmbeddingNotConfigured, MemoryIndexPending, MemorySummaryStale
from .hierarchy import MemoryHierarchy
from .maintenance import MemoryCleanupWorker
from .policy import MemoryPolicy
from .ports import MemoryVectorIndex
from .repository import JsonMemoryRepository
from .retention import MemoryRetentionPolicy
from .retrieval import HybridMemoryRetriever, LexicalMemoryRetriever
from .types import (
    MemoryCleanupPlan, MemoryCleanupReport, MemoryContext, MemoryHit, MemoryIndexEvent, MemoryLayer, MemoryMutation, MemoryQuery,
    MemoryRecord, MemorySnapshot, MemorySourceRef, MemorySyncReport, SearchMode,
)
from .writer import MemoryWriter, message_evidence_id, validate_mutation


class MemoryService:
    def __init__(
        self, path: Path, user_id: str = "default", *, config: MemoryConfig | None = None,
        embedding: MemoryVectorIndex | None = None, summary_model: ChatModel | None = None,
        token_counter: Any = None, repository: JsonMemoryRepository | None = None,
        clock: Callable[[], datetime] = utc_now,
    ):
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id 必须为非空字符串")
        self.config = config if config is not None else MemoryConfig()
        self.repository = repository if repository is not None else JsonMemoryRepository(path, user_id)
        if self.repository.user_id != user_id or self.repository.path.resolve() != Path(path).resolve():
            raise ValueError("注入的 repository 与当前用户/业务文件不一致")
        if embedding is not None and embedding.user_id != user_id:
            raise ValueError("注入的记忆向量桥接器属于其他用户")
        self.user_id = user_id
        self.path = Path(path)
        self.embedding = embedding
        self.clock = clock
        self.policy = MemoryPolicy()
        self.lexical = LexicalMemoryRetriever(self.config)
        self.retriever = HybridMemoryRetriever(self.lexical, embedding, self.config)
        self.hierarchy = MemoryHierarchy(self.config, summary_model, token_counter)
        self.writer = MemoryWriter(summary_model)
        self.last_search_hits: list[MemoryHit] = []
        self.last_sync_report = MemorySyncReport()
        self.retention = MemoryRetentionPolicy(self.config)
        self.last_cleanup_report: MemoryCleanupReport | None = None
        self.history_cleanup_worker = MemoryCleanupWorker(self)

    def add(
        self, content: str, memory_type: str = "episodic", importance: float = 0.5,
        metadata: dict[str, Any] | None = None, *, layer: MemoryLayer = "archive",
        task_id: str | None = None, valid_from: datetime | None = None,
        valid_to: datetime | None = None, fact_key: str | None = None, memory_id: str | None = None,
    ) -> MemoryRecord:
        """构造 MemoryRecord 和 upsert 事件，一次提交后返回副本。

        读取最新快照→生成安全 ID→构造记录→保存成功才返回。
        now=self.clock()；created_at/updated_at 均取 now，version=1，不改其他记录时间。
        layer=working 要有 task_id；summary 由 finish_task 生成并保留 source_refs。
        同用户同 fact_key 的有效区间不能重叠，变更已有事实应走 revise_fact。
        不在 add 内请求嵌入或生成摘要；后续 flush/process 显式执行。
        """
        if layer == "summary":
            raise ValueError("摘要由 finish_task 生成，普通 add 不接受 summary 层")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("metadata 必须为字典或 None")
        now = self.clock()
        memory_id = f"memory_{uuid4().hex}" if memory_id is None else memory_id
        event_id = f"event_{uuid4().hex}"
        metadata = deepcopy(metadata) if metadata is not None else {}
        new_memory = MemoryRecord(
            id=memory_id,
            content=content,
            user_id=self.user_id,
            memory_type=memory_type,
            importance=importance,
            created_at=now,
            updated_at=now,
            metadata=metadata,
            layer=layer,
            valid_from=valid_from,
            valid_to=valid_to,
            task_id=task_id,
            version=1,
            fact_key=fact_key,
        )
        new_event = MemoryIndexEvent(
            id=event_id, user_id=self.user_id, memory_id=memory_id,
            operation="upsert", version=new_memory.version, created_at=now,
        )
        # 复用已完成的字段/时间/元数据校验；这里只检查，不写文件或生成向量。
        self.repository.codec.encode(MemorySnapshot(records=[new_memory], index_events=[new_event]))

        for attempt in range(3):
            snapshot = self.repository.read_snapshot()
            existing_id = next((record for record in snapshot.records if record.id == memory_id), None)
            if existing_id is not None:
                if all(getattr(existing_id, field) == getattr(new_memory, field) for field in (
                    "content", "memory_type", "importance", "metadata", "layer", "task_id", "valid_from", "valid_to", "fact_key",
                )):
                    return existing_id
                raise MemoryConflict("记忆 ID 已对应其他内容")
            if fact_key is not None:
                for existing in snapshot.records:
                    if existing.fact_key != fact_key:
                        continue
                    # [from,to) 相邻不重叠；None 表示该侧没有界限。
                    starts_before_end = (
                        valid_to is None or existing.valid_from is None or existing.valid_from < valid_to
                    )
                    ends_after_start = (
                        valid_from is None or existing.valid_to is None or valid_from < existing.valid_to
                    )
                    if starts_before_end and ends_after_start:
                        raise ValueError("同一 fact_key 的有效期不能重叠，更新已有事实请使用 revise_fact")
            snapshot.records.append(new_memory)
            snapshot.index_events.append(new_event)
            try:
                saved_snapshot = self.repository.commit(snapshot)
            except MemoryConflict:
                if attempt == 2:
                    raise
                continue
            return next(item for item in saved_snapshot.records if item.id == memory_id)

    def get(self, memory_id: str) -> MemoryRecord:
        """只返回当前用户 ID 对应记录的副本，不存在抛 ValueError。"""
        if not isinstance(memory_id, str) or not memory_id.strip():
            raise ValueError("memory_id 必须是非空字符串")
        snapshot = self.repository.read_snapshot()
        for record in snapshot.records:
            if record.id == memory_id:
                return deepcopy(record)
        raise ValueError(f"找不到记忆 {memory_id}")

    def update(
        self, memory_id: str, content: str | None = None, importance: float | None = None,
        *, metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        """保留 ID/归属/创建时间，更新 updated_at 和记录 version。

        None 表示不修改，importance=0 必须生效；metadata 按整体替换约定复制。
        有 fact_key 的事实不能直接改正文，使用 revise_fact 保留旧版本。
        原始记录变化时，引用旧版本的摘要置 stale，追加其索引删除事件。
        当前记录 upsert 与所有相关变更在同一 JSON 提交；不请求模型。
        未传修改字段或值未改变时返回副本，不消耗版本或生成事件。
        """
        if not isinstance(memory_id, str) or not memory_id.strip():
            raise ValueError("memory_id 必须是非空字符串")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("metadata 必须为字典或 None")
        changes = {
            "content": content,
            "importance": importance,
            "metadata": deepcopy(metadata) if metadata is not None else None,
        }
        now = self.clock()
        event_id = f"event_{uuid4().hex}"
        delete_event_ids = {}
        for attempt in range(3):
            snapshot = self.repository.read_snapshot()
            memory_to_update = None
            for record in snapshot.records:
                if record.id == memory_id:
                    memory_to_update = record
                    break
            if memory_to_update is None:
                raise ValueError(f"找不到记忆 {memory_id}")
            if memory_to_update.fact_key is not None and content is not None and content != memory_to_update.content:
                raise ValueError("修改事实正文请使用 revise_fact，以保留历史有效区间")

            has_changes = False
            for key, value in changes.items():
                if value is not None:
                    if getattr(memory_to_update, key) != value:
                        has_changes = True
                    setattr(memory_to_update, key, deepcopy(value))
            # 先校验再判断是否有变化，避免把 False 当成合法的 importance=0。
            self.repository.codec.encode(MemorySnapshot(records=[memory_to_update]))
            if not has_changes:
                return deepcopy(memory_to_update)

            memory_to_update.updated_at = now
            memory_to_update.version += 1
            event = MemoryIndexEvent(
                id=event_id, user_id=self.user_id, memory_id=memory_id,
                operation="upsert", version=memory_to_update.version, created_at=now,
            )
            snapshot.index_events.append(event)
            for summary in snapshot.records:
                if summary.layer != "summary" or summary.summary_state == "stale":
                    continue
                if any(ref.memory_id == memory_id and ref.version != memory_to_update.version
                       for ref in summary.source_refs):
                    summary.summary_state = "stale"
                    summary.updated_at = now
                    summary.version += 1
                    if summary.id not in delete_event_ids:
                        delete_event_ids[summary.id] = f"event_{uuid4().hex}"
                    snapshot.index_events.append(MemoryIndexEvent(
                        id=delete_event_ids[summary.id], user_id=self.user_id,
                        memory_id=summary.id, operation="delete", version=summary.version, created_at=now,
                    ))
            try:
                saved_snapshot = self.repository.commit(snapshot)
            except MemoryConflict:
                if attempt == 2:
                    raise
                continue
            return next(item for item in saved_snapshot.records if item.id == memory_id)

    def remove(self, memory_id: str) -> bool:
        """删除当前用户记录并写 delete 事件，保留其他用户。

        不存在/跨用户返回 False，I/O 错误抛出；删除源记录也要使摘要失效。
        删除向量检索成员不等于删除共享缓存；不要先删事实再单独尝试写事件。
        """
        if not isinstance(memory_id, str) or not memory_id.strip():
            raise ValueError("memory_id 必须是非空字符串")
        event_id = f"event_{uuid4().hex}"
        delete_event_ids = {}
        now = None
        for attempt in range(3):
            snapshot = self.repository.read_snapshot()
            memory_to_delete = None
            new_records = []
            for record in snapshot.records:
                if record.id == memory_id:
                    memory_to_delete = record
                else:
                    new_records.append(record)
            if memory_to_delete is None or memory_to_delete.user_id != self.user_id:
                return False
            if now is None:
                now = self.clock()
            snapshot.index_events.append(MemoryIndexEvent(
                id=event_id, user_id=self.user_id, memory_id=memory_id,
                operation="delete", version=memory_to_delete.version + 1, created_at=now,
            ))
            for summary in new_records:
                if summary.layer != "summary" or summary.summary_state == "stale":
                    continue
                if any(ref.memory_id == memory_id for ref in summary.source_refs):
                    summary.summary_state = "stale"
                    summary.updated_at = now
                    summary.version += 1
                    if summary.id not in delete_event_ids:
                        delete_event_ids[summary.id] = f"event_{uuid4().hex}"
                    snapshot.index_events.append(MemoryIndexEvent(
                        id=delete_event_ids[summary.id], user_id=self.user_id,
                        memory_id=summary.id, operation="delete", version=summary.version, created_at=now,
                    ))
            snapshot.records = new_records
            try:
                self.repository.commit(snapshot)
            except MemoryConflict:
                if attempt == 2:
                    raise
                continue
            return True

    def revise_fact(self, memory_id: str, content: str, effective_from: datetime) -> MemoryRecord:
        """关闭旧事实区间，创建新 ID 的事实版本，二者一次提交。

        同 user/fact_key；旧 valid_to=effective_from，新 valid_from=effective_from。
        新记录 supersedes_id 指向旧 ID，旧正文保留供 as_of 检索。
        目标必须有 fact_key；生效时间严格位于旧有效区间内部。
        新记录 valid_to 继承旧记录原来的结束时间，只拆分原区间，不覆盖后续事实。
        创建/更新时间取 clock，不把回填的 effective_from 冒充记录时间。
        version 是单条记录修订号，不是整个事实链编号；不能覆盖已有事实历史。
        更新索引元数据、使受影响摘要失效，全部事件与记录一起保存。
        """
        if not isinstance(memory_id, str) or not memory_id.strip():
            raise ValueError("memory_id 必须是非空字符串")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content 必须是非空字符串")
        if not isinstance(effective_from, datetime) or effective_from.utcoffset() is None:
            raise ValueError("effective_from 必须是带时区的 datetime")
        new_id = f"memory_{uuid4().hex}"
        old_event_id = f"event_{uuid4().hex}"
        new_event_id = f"event_{uuid4().hex}"
        delete_event_ids = {}
        now = None
        for attempt in range(3):
            snapshot = self.repository.read_snapshot()
            fact_memory = next((record for record in snapshot.records if record.id == memory_id), None)
            if fact_memory is None:
                raise ValueError(f"找不到记忆 {memory_id}")
            if fact_memory.fact_key is None:
                raise ValueError("revise_fact 只能修订带 fact_key 的事实")
            if (fact_memory.valid_from is not None and effective_from <= fact_memory.valid_from
                    or fact_memory.valid_to is not None and effective_from >= fact_memory.valid_to):
                raise ValueError("effective_from 必须严格位于旧事实的有效区间内部")
            if now is None:
                now = self.clock()
            old_end = fact_memory.valid_to
            fact_memory.valid_to = effective_from
            fact_memory.version += 1
            fact_memory.updated_at = now
            new_fact = MemoryRecord(
                id=new_id, content=content, user_id=self.user_id,
                memory_type=fact_memory.memory_type, importance=fact_memory.importance,
                created_at=now, updated_at=now, metadata=deepcopy(fact_memory.metadata),
                layer=fact_memory.layer, valid_from=effective_from, valid_to=old_end,
                task_id=fact_memory.task_id, version=1, fact_key=fact_memory.fact_key,
                supersedes_id=fact_memory.id,
            )
            snapshot.index_events.extend([
                MemoryIndexEvent(old_event_id, self.user_id, fact_memory.id, "upsert", fact_memory.version, now),
                MemoryIndexEvent(new_event_id, self.user_id, new_fact.id, "upsert", new_fact.version, now),
            ])
            snapshot.records.append(new_fact)
            for summary in snapshot.records:
                if summary.layer != "summary" or summary.summary_state == "stale":
                    continue
                if any(ref.memory_id == memory_id and ref.version != fact_memory.version
                       for ref in summary.source_refs):
                    summary.summary_state = "stale"
                    summary.updated_at = now
                    summary.version += 1
                    if summary.id not in delete_event_ids:
                        delete_event_ids[summary.id] = f"event_{uuid4().hex}"
                    snapshot.index_events.append(MemoryIndexEvent(
                        delete_event_ids[summary.id], self.user_id, summary.id, "delete", summary.version, now,
                    ))
            try:
                saved_snapshot = self.repository.commit(snapshot)
            except MemoryConflict:
                if attempt == 2:
                    raise
                continue
            return next(record for record in saved_snapshot.records if record.id == new_id)

    def list_records(self, filters: MemoryQuery | None = None) -> list[MemoryRecord]:
        """读取当前用户快照并按条件筛选；默认查当前有效记录，返回独立副本。"""
        snapshot = self.repository.read_snapshot()
        if filters is None:
            filters = MemoryQuery()
        now = self.clock()
        results = self.policy.select(snapshot.records, self.user_id, filters, now)
        return results

    def plan_history_cleanup(self, limit: int | None = None) -> MemoryCleanupPlan:
        """已提供：读取当前用户快照，交给 M08 策略预览；本入口不写文件。"""
        if limit is not None and (type(limit) is not int or limit < 0):
            raise ValueError("limit 必须为 None 或非负整数")
        snapshot = self.repository.read_snapshot()
        return self.retention.plan(snapshot, self.user_id, self.clock(), limit)

    def cleanup_history(self, dry_run: bool = True, limit: int | None = None) -> MemoryCleanupReport:
        """预览或按策略批量删除 archive 记忆，并原子保存 delete 待办。

        本方法手动调用不依赖 auto_cleanup_history，默认 dry_run=True。
        读取最新快照，用 retention.plan 重新计算；预览不写文件，报告只列候选。
        实际执行按候选 ID/version 删除当前用户记录，为每条追加 delete 索引事件，
        版本为被删除记录 version+1；单次 commit，保存成功后才填写 deleted_ids、
        index_event_ids 和 committed_revision，再更新 last_cleanup_report。
        无候选时不提交、不增加文件 revision。失败抛出，不报告虚构成功。
        MemoryConflict 最多尝试 3 次：每次重读、重新判定置顶/摘要引用保护与版本，
        不能直接重放旧候选，避免删除已被延长有效期、最近修改、置顶或新摘要引用的记录。
        两类起算时间交给 retention.plan 统一判断，不另写只接受 fact 的删除分支。
        保留其他用户、现有待办和剩余事实的 supersedes_id；允许历史链被截断。
        不循环调用 remove 形成半批提交，不直接删除共享 vectors 缓存或调用模型。
        """
        if type(dry_run) is not bool:
            raise ValueError("dry_run 必须是 bool")
        if limit is not None and (type(limit) is not int or limit < 0):
            raise ValueError("limit 必须为 None 或非负整数")
        event_ids: dict[str, str] = {}
        for attempt in range(3):
            snapshot = self.repository.read_snapshot()
            cleanup_plan = self.retention.plan(snapshot, self.user_id, self.clock(), limit)
            candidates = cleanup_plan.candidates
            report = MemoryCleanupReport(plan=cleanup_plan, dry_run=dry_run)
            if dry_run or not candidates:
                self.last_cleanup_report = report
                return report

            # 按计划顺序生成事件和报告；文件中的记录顺序不代表清理优先级。
            by_id = {record.id: record for record in snapshot.records}
            target_versions = {ref.memory_id: ref.version for ref in candidates}
            if len(target_versions) != len(candidates) or any(
                ref.memory_id not in by_id or by_id[ref.memory_id].version != ref.version
                for ref in candidates
            ):
                raise ValueError("清理候选与当前快照不一致")

            deleted_ids = []
            index_event_ids = []
            for ref in candidates:
                record = by_id[ref.memory_id]
                event_id = event_ids.get(record.id)
                if event_id is None:
                    event_id = event_ids[record.id] = f"event_{uuid4().hex}"
                snapshot.index_events.append(MemoryIndexEvent(
                    id=event_id, user_id=self.user_id, memory_id=record.id,
                    operation="delete", version=record.version + 1,
                    created_at=cleanup_plan.checked_at,
                ))
                deleted_ids.append(record.id)
                index_event_ids.append(event_id)
            snapshot.records = [record for record in snapshot.records if record.id not in target_versions]
            try:
                saved_snapshot = self.repository.commit(snapshot)
            except MemoryConflict:
                if attempt == 2:
                    raise
                continue
            report.committed_revision = saved_snapshot.revision
            report.deleted_ids = deleted_ids
            report.index_event_ids = index_event_ids
            self.last_cleanup_report = report
            return report

    def search_hits(
        self, query: str, limit: int = 5, *, mode: SearchMode | None = None,
        filters: MemoryQuery | None = None,
    ) -> list[MemoryHit]:
        """先筛选当前用户的记忆，再返回带分数和名次的检索命中。

        mode 省略时使用配置值。当前用户的快照中存在待同步事件时，
        向量和混合模式抛出 MemoryIndexPending。成功后更新 last_search_hits。
        """
        mode = mode if mode is not None else self.config.search_mode
        if mode not in ("lexical", "vector", "hybrid"):
            raise ValueError("不支持的记忆检索模式")
        if type(limit) is not int or limit < 0:
            raise ValueError("limit 必须是非负整数")
        if not isinstance(query, str):
            raise ValueError("query 必须是字符串")
        filters = filters if filters is not None else MemoryQuery()
        snapshot = self.repository.read_snapshot()
        if mode in ("vector", "hybrid") and snapshot.index_events:
            raise MemoryIndexPending("当前用户的记忆索引仍有待同步事件")
        records = snapshot.records
        candidates = self.policy.select(records, self.user_id, filters, self.clock())
        results = self.retriever.search(query, candidates, limit, mode)
        self.last_search_hits = results
        return results

    def search(
        self, query: str, limit: int = 5, *, mode: SearchMode | None = None,
        as_of: datetime | None = None, layers: tuple[MemoryLayer, ...] | None = None,
        task_id: str | None = None, memory_type: str | None = None,
        min_importance: float | None = None, max_importance: float | None = None,
    ) -> list[MemoryRecord]:
        """已提供的查询入口；返回 MemoryRecord，供 MemoryTool/ContextBuilder 使用。"""
        filters = MemoryQuery(as_of, layers, task_id, memory_type, min_importance, max_importance)
        return [hit.memory for hit in self.search_hits(query, limit, mode=mode, filters=filters)]

    def flush_index_events(self, limit: int = 100) -> MemorySyncReport:
        """按最新业务状态同步至多 limit 个来源，并返回同步报告。

        每个来源合并处理本轮读到的事件。同步成功后确认这些事件的 ID，
        失败时保留事件并在报告中记录错误码。向量生成由 process_pending 负责。
        """
        if type(limit) is not int or limit < 0:
            raise ValueError("limit 必须是非负整数")

        # 同一业务文件的同步调用依次执行，业务提交可以继续使用独立的锁。
        with self.repository.locked(index_sync=True):
            report = MemorySyncReport()
            snapshot = self.repository.read_snapshot()
            events_by_id: dict[str, list[MemoryIndexEvent]] = {}
            for event in snapshot.index_events:
                events_by_id.setdefault(event.memory_id, []).append(event)

            groups = list(events_by_id.items())[:limit]
            if not groups:
                report.pending = len(events_by_id)
                self.last_sync_report = report
                return report

            embedding = self.embedding
            if embedding is None:
                raise MemoryEmbeddingNotConfigured("尚未注入记忆嵌入桥接器")

            records_by_id = {record.id: record for record in snapshot.records}
            completed_event_ids: list[str] = []
            for memory_id, events in groups:
                record = records_by_id.get(memory_id)
                try:
                    embedding.apply(memory_id, record)
                except ExercisePending:
                    raise
                except Exception:
                    report.errors[memory_id] = "index_sync_failed"
                    continue
                completed_event_ids.extend(event.id for event in events)
                report.applied += 1

            if completed_event_ids:
                self.repository.acknowledge(completed_event_ids)
            snapshot = self.repository.read_snapshot()
            report.pending = len({event.memory_id for event in snapshot.index_events})
            self.last_sync_report = report
            return report

    def process_pending(self, limit: int | None = None) -> int:
        """已提供的显式执行委托；不会隐式刷新尚未完成的业务 outbox。"""
        if self.embedding is None:
            raise MemoryEmbeddingNotConfigured("尚未注入记忆嵌入桥接器")
        return self.embedding.process_pending(limit)

    def rebuild_index(self) -> int:
        """为当前用户的已存记忆追加索引待办，返回本次追加的数量。

        失效摘要安排移除索引，其余记忆安排同步正文。已有待办和业务记录
        一并保留。快照冲突时重读并重建待办，最多尝试提交三次。
        """
        for attempt in range(3):
            snapshot = self.repository.read_snapshot()
            if not snapshot.records:
                return 0

            now = self.clock()
            events = []
            for record in snapshot.records:
                is_stale_summary = record.layer == "summary" and record.summary_state == "stale"
                events.append(MemoryIndexEvent(
                    id=f"event_{uuid4().hex}",
                    user_id=self.user_id,
                    memory_id=record.id,
                    operation="delete" if is_stale_summary else "upsert",
                    version=record.version,
                    created_at=now,
                ))
            snapshot.index_events.extend(events)
            try:
                self.repository.commit(snapshot)
            except MemoryConflict:
                if attempt == 2:
                    raise
                continue
            return len(events)

    def checkpoint_task(self, request: HistorySummaryRequest) -> MemoryRecord:
        """验证历史前缀，原子保存原始消息、阶段摘要与索引待办。"""
        if not isinstance(request, HistorySummaryRequest):
            raise ValueError("request 必须是 HistorySummaryRequest")
        request = deepcopy(request)
        if not all(isinstance(value, str) and value.strip() for value in (request.history_id, request.task_id)):
            raise ValueError("history_id 和 task_id 必须是非空字符串")
        start, end = request.start_index, request.end_index
        if (type(start) is not int or type(end) is not int or not 0 <= start < end
                or not isinstance(request.messages, list) or len(request.messages) != end - start):
            raise ValueError("阶段摘要的消息范围无效")
        previous_id = request.previous_summary_id
        if (start == 0) != (previous_id is None) or (
            previous_id is not None and (not isinstance(previous_id, str) or not previous_id.strip())
        ):
            raise ValueError("历史范围与前一条阶段摘要不匹配")
        if request.target_tokens is not None and (type(request.target_tokens) is not int or request.target_tokens <= 0):
            raise ValueError("阶段摘要预算必须是正整数")
        incoming_hash = history_prefix_hash(request.messages)
        if any(not message.content.strip() for message in request.messages):
            raise ValueError("归档的历史消息正文必须非空")
        if start == 0 and incoming_hash != request.prefix_hash:
            raise ValueError("阶段摘要的历史指纹不匹配")

        draft = None
        draft_previous_version = None
        for attempt in range(3):
            snapshot = self.repository.read_snapshot()
            by_id = {record.id: record for record in snapshot.records}
            active = [
                record for record in snapshot.records
                if record.layer == "summary" and record.summary_state == "fresh"
                and record.task_id == request.task_id and record.metadata.get("summary_kind") == "progress"
                and record.metadata.get("history_id") == request.history_id
            ]
            if len(active) > 1:
                raise MemoryConflict("同一历史存在多条当前阶段摘要")
            previous = active[0] if active else None
            if previous is not None and previous.metadata.get("covered_until") == end:
                if previous.metadata.get("prefix_hash") != request.prefix_hash:
                    raise MemoryConflict("同一历史范围已经按不同内容摘要")
                saved_messages = self._checkpoint_history(previous, snapshot.records)
                if history_prefix_hash(saved_messages[start:end]) != incoming_hash:
                    raise ValueError("重放请求的原始消息与已保存的历史不匹配")
                return previous

            prefix = []
            if previous_id is not None:
                if (previous is None or previous.id != previous_id
                        or previous.metadata.get("covered_until") != start):
                    raise MemorySummaryStale("前一条阶段摘要已失效或覆盖范围不匹配")
                prefix = self._checkpoint_history(previous, snapshot.records)
            elif previous is not None:
                raise MemoryConflict("当前历史已保存阶段摘要")
            if history_prefix_hash(prefix + request.messages) != request.prefix_hash:
                raise ValueError("阶段摘要的历史指纹不匹配")

            now = self.clock()
            sources = []
            for index, message in enumerate(request.messages, start=start):
                key = json.dumps([self.user_id, request.task_id, request.history_id, index], ensure_ascii=False)
                memory_id = f"history_{uuid5(NAMESPACE_URL, key).hex}"
                if memory_id in by_id:
                    raise MemoryConflict("归档的历史位置已被占用")
                sources.append(MemoryRecord(
                    id=memory_id, content=message.content, user_id=self.user_id,
                    memory_type="conversation", layer="archive", task_id=request.task_id,
                    created_at=message.timestamp, updated_at=max(now, message.timestamp),
                    metadata={"history_id": request.history_id, "message_index": index, "role": message.role},
                ))
            previous_version = previous.version if previous is not None else None
            if draft is None or draft_previous_version != previous_version:
                draft = self.hierarchy.summarize_progress(
                    sources, request.task_id, self.user_id, now,
                    previous_summary=previous, max_tokens=request.target_tokens,
                )
                draft_previous_version = previous_version
            draft.metadata.update(history_id=request.history_id, covered_until=end, prefix_hash=request.prefix_hash)
            draft.created_at = draft.updated_at = now
            snapshot.records.extend([*sources, draft])
            snapshot.index_events.extend(
                MemoryIndexEvent(f"event_{uuid4().hex}", self.user_id, record.id, "upsert", record.version, now)
                for record in [*sources, draft]
            )
            if previous is not None:
                previous.summary_state = "stale"
                previous.version += 1
                previous.updated_at = now
                snapshot.index_events.append(MemoryIndexEvent(
                    f"event_{uuid4().hex}", self.user_id, previous.id, "delete", previous.version, now,
                ))
            try:
                saved = self.repository.commit(snapshot)
            except MemoryConflict:
                if attempt == 2:
                    raise
                continue
            return next(record for record in saved.records if record.id == draft.id)

    def _checkpoint_history(self, summary: MemoryRecord, records: list[MemoryRecord]) -> list[Message]:
        """从已保存的阶段摘要还原并核对其完整原始前缀。"""
        messages = []
        for index, source in enumerate(self.hierarchy.expand(summary, records)):
            if (source.layer != "archive" or source.task_id != summary.task_id
                    or source.metadata.get("history_id") != summary.metadata.get("history_id")
                    or source.metadata.get("message_index") != index
                    or not isinstance(source.metadata.get("role"), str)):
                raise MemorySummaryStale("阶段摘要的原始历史位置不匹配")
            messages.append(Message(source.metadata["role"], source.content, source.created_at))
        if (len(messages) != summary.metadata.get("covered_until")
                or history_prefix_hash(messages) != summary.metadata.get("prefix_hash")):
            raise MemorySummaryStale("阶段摘要的原始历史指纹不匹配")
        return messages

    def finish_task(self, task_id: str) -> MemoryRecord:
        """归档当前任务的工作记忆，原子保存摘要和索引待办。

        摘要引用归档后的原文版本。重复完成未变的任务时复用已有摘要。
        提交冲突时重新读取快照，草稿的全部来源版本匹配时可复用模型结果。
        """
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("task_id 必须为非空字符串")

        draft = None
        for attempt in range(3):
            snapshot = self.repository.read_snapshot()
            archived_records = [
                record for record in snapshot.records
                if record.task_id == task_id and record.layer == "working"
            ]
            if not archived_records:
                previous = next((
                    record for record in reversed(snapshot.records)
                    if record.task_id == task_id and record.layer == "summary"
                    and record.metadata.get("summary_kind") != "progress"
                ), None)
                if previous is None:
                    raise ValueError(f"任务 {task_id} 没有待归档的工作记忆")
                if not previous.source_refs:
                    raise MemorySummaryStale(f"任务 {task_id} 的摘要缺少原始记录引用")
                self.hierarchy.expand(previous, snapshot.records)
                return previous

            now = self.clock()
            for record in archived_records:
                record.layer = "archive"
                record.updated_at = now
                record.version += 1

            archived_versions = {record.id: record.version for record in archived_records}
            source_refs = [MemorySourceRef(memory_id, version) for memory_id, version in archived_versions.items()]
            if draft is None or draft.source_refs != source_refs:
                draft = self.hierarchy.summarize(archived_records, task_id, self.user_id, now)
            else:
                draft.created_at = now
                draft.updated_at = now

            stale_summaries = []
            for record in snapshot.records:
                if record.layer != "summary" or record.summary_state == "stale":
                    continue
                if any(ref.memory_id in archived_versions and ref.version != archived_versions[ref.memory_id]
                       for ref in record.source_refs):
                    record.summary_state = "stale"
                    record.updated_at = now
                    record.version += 1
                    stale_summaries.append(record)

            snapshot.records.append(draft)
            for operation, changed_records in (("upsert", [*archived_records, draft]), ("delete", stale_summaries)):
                for record in changed_records:
                    snapshot.index_events.append(MemoryIndexEvent(
                        id=f"event_{uuid4().hex}", user_id=self.user_id, memory_id=record.id,
                        operation=operation, version=record.version, created_at=now,
                    ))

            try:
                saved_snapshot = self.repository.commit(snapshot)
            except MemoryConflict:
                if attempt == 2:
                    raise
                continue
            return next(record for record in saved_snapshot.records if record.id == draft.id)

    def expand_summary(self, summary_id: str) -> list[MemoryRecord]:
        """在同一份用户快照中读取摘要和它引用的原始记录副本。

        找不到摘要 ID 时抛出 ValueError，来源版本和摘要状态由 expand 核对。
        """
        if not isinstance(summary_id, str) or not summary_id.strip():
            raise ValueError("summary_id 必须为非空字符串")
        snapshot = self.repository.read_snapshot()
        for record in snapshot.records:
            if record.id == summary_id:
                return self.hierarchy.expand(record, snapshot.records)
        raise ValueError(f"找不到摘要 {summary_id}")

    def rebuild_summary(self, summary_id: str) -> MemoryRecord:
        """用当前版本的原始记录重建摘要，保留摘要 ID 和创建时间。

        摘要和索引待办一次提交。来源缺失时抛出 MemorySummaryStale，
        已有摘要仍有效时直接返回。提交冲突后重读快照并重新核对来源版本。
        """
        if not isinstance(summary_id, str) or not summary_id.strip():
            raise ValueError("summary_id 必须为非空字符串")

        draft = None
        for attempt in range(3):
            snapshot = self.repository.read_snapshot()
            record_by_id = {record.id: record for record in snapshot.records}
            summary = record_by_id.get(summary_id)
            if summary is None:
                raise ValueError(f"找不到摘要 {summary_id}")
            if summary.layer != "summary":
                raise ValueError("只能重建 summary 层的记忆")
            if not summary.source_refs:
                raise MemorySummaryStale(f"摘要 {summary_id} 缺少原始记录引用")

            ref_records = []
            for ref in summary.source_refs:
                source = record_by_id.get(ref.memory_id)
                if source is None:
                    raise MemorySummaryStale(f"找不到摘要来源 {ref.memory_id}")
                ref_records.append(source)
            current_refs = [MemorySourceRef(source.id, source.version) for source in ref_records]
            if summary.summary_state == "fresh" and summary.source_refs == current_refs:
                return summary

            now = self.clock()
            if draft is None or draft.task_id != summary.task_id or draft.source_refs != current_refs:
                draft = self.hierarchy.summarize(ref_records, summary.task_id, self.user_id, now)
            summary.content = draft.content
            summary.importance = draft.importance
            summary.source_refs = draft.source_refs
            summary.metadata.update(draft.metadata)
            summary.updated_at = now
            summary.version += 1
            summary.summary_state = "fresh"
            snapshot.index_events.append(MemoryIndexEvent(
                id=f"event_{uuid4().hex}",
                user_id=self.user_id,
                memory_id=summary.id,
                operation="upsert",
                version=summary.version,
                created_at=now,
            ))

            try:
                saved_snapshot = self.repository.commit(snapshot)
            except MemoryConflict:
                if attempt == 2:
                    raise
                continue
            return next(record for record in saved_snapshot.records if record.id == summary_id)

    def prepare_context(
        self, query: str, task_id: str, max_tokens: int | None = None, as_of: datetime | None = None,
    ) -> MemoryContext:
        """按本轮预算组织当前任务记忆、相关摘要、有效事实与归档证据。

        所有候选来自同一用户快照。当前任务的工作记忆独立于查询匹配。
        从排名最高且带引用的摘要中取至多两条符合本轮有效期的归档证据。
        执行向量或混合检索前检查当前快照的索引待办。
        """
        if not isinstance(query, str):
            raise ValueError("query 必须是字符串")
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("task_id 必须为非空字符串")

        snapshot = self.repository.read_snapshot()
        records = snapshot.records
        candidates = self.policy.select(records, self.user_id, MemoryQuery(as_of=as_of), self.clock())
        working = []
        other_candidates = []
        for record in candidates:
            if record.task_id == task_id and record.layer == "working":
                working.append(record)
            elif record.layer == "summary" or (record.layer == "archive" and record.fact_key is not None):
                other_candidates.append(record)

        ranked_memories = []
        if query.strip() and other_candidates:
            mode = self.config.search_mode
            if mode in ("vector", "hybrid") and snapshot.index_events:
                raise MemoryIndexPending("当前用户的记忆索引仍有待同步事件")
            hits = self.retriever.search(query, other_candidates, len(other_candidates), mode)
            ranked_memories = [hit.memory for hit in hits]

        budget = self.config.context_tokens if max_tokens is None else max_tokens
        top_summary = next(
            (memory for memory in ranked_memories if memory.layer == "summary" and memory.source_refs),
            None,
        )
        archive = []
        if top_summary is not None:
            eligible_ids = {
                record.id for record in candidates
                if record.layer == "archive" and record.fact_key is None
            }
            for record in self.hierarchy.expand(top_summary, records):
                if record.id in eligible_ids:
                    archive.append(record)
                    eligible_ids.remove(record.id)
                    if len(archive) == 2:
                        break

        return self.hierarchy.build_context(
            working + ranked_memories + archive, task_id, self.user_id, budget,
        )

    def propose(self, messages: list[Message]) -> list[MemoryMutation]:
        """根据当前用户的对话和相关有效记忆生成有证据的写入建议。"""
        if not isinstance(messages, list):
            raise ValueError("messages 必须是列表")
        if not messages:
            return []
        history_prefix_hash(messages)
        snapshot = self.repository.read_snapshot()
        eligible = self.policy.select(snapshot.records, self.user_id, MemoryQuery(), self.clock())
        candidates = [record for record in eligible if record.layer != "summary"]
        query = "\n".join(message.content for message in messages)
        related = {hit.memory.id: hit.memory for hit in self.lexical.search(query, candidates, 20)}
        for record in sorted(candidates, key=lambda item: item.updated_at, reverse=True):
            if len(related) >= 20:
                break
            if record.fact_key is not None:
                related.setdefault(record.id, record)
        return self.writer.propose(messages, list(related.values()))

    def apply_mutations(self, mutations: list[MemoryMutation]) -> list[MemoryRecord]:
        """核对证据和目标版本，在一次提交中应用整批写入建议。"""
        if not isinstance(mutations, list):
            raise ValueError("mutations 必须是列表")
        if not mutations:
            return []
        mutations = deepcopy(mutations)
        for mutation in mutations:
            validate_mutation(mutation)

        for attempt in range(3):
            snapshot = self.repository.read_snapshot()
            by_id = {record.id: record for record in snapshot.records}
            evidence_versions = {record.id: record.version for record in snapshot.records}
            changed_targets = set()
            for mutation in mutations:
                for evidence_id in mutation.evidence_ids:
                    if evidence_id in by_id:
                        continue
                    evidence = mutation.evidence_messages.get(evidence_id)
                    if evidence is None or message_evidence_id(evidence) != evidence_id:
                        raise ValueError("记忆写入建议引用了未知证据")
                if mutation.target_id is None:
                    continue
                target = by_id.get(mutation.target_id)
                if target is None or target.version != mutation.expected_version:
                    raise MemoryConflict("记忆写入建议的目标或版本已变化")
                if mutation.fact_key is not None and mutation.fact_key != target.fact_key:
                    raise ValueError("建议中的事实键与目标不匹配")
                if mutation.operation != "noop":
                    if target.layer == "summary" or target.id in changed_targets:
                        raise ValueError("批次包含摘要目标或重复修改目标")
                    changed_targets.add(target.id)

            now = self.clock()
            upserts = []
            returned_ids = []
            for mutation in mutations:
                operation = mutation.operation
                if operation == "noop":
                    continue
                if operation == "add":
                    fact_key = mutation.fact_key
                    effective = mutation.effective_from
                    existing = [
                        record for record in snapshot.records
                        if record.layer != "summary" and record.fact_key == fact_key
                    ]
                    at = effective if effective is not None else now
                    if any(record.content == mutation.content
                           and (record.valid_from is None or record.valid_from <= at)
                           and (record.valid_to is None or at < record.valid_to) for record in existing):
                        continue
                    if fact_key is not None and any(
                        effective is None or record.valid_to is None or effective < record.valid_to
                        for record in existing
                    ):
                        raise ValueError("同一事实的有效期重叠，请用 revise")
                    item = MemoryRecord(
                        id=f"memory_{uuid4().hex}", user_id=self.user_id, content=mutation.content,
                        memory_type="semantic" if fact_key is not None else "episodic",
                        created_at=now, updated_at=now, fact_key=fact_key, valid_from=effective,
                    )
                    snapshot.records.append(item)
                else:
                    target = by_id[mutation.target_id]
                    if operation == "update":
                        if mutation.content == target.content:
                            continue
                        if target.fact_key is not None:
                            raise ValueError("修改事实正文请使用 revise")
                        target.content = mutation.content
                        target.updated_at = now
                        target.version += 1
                        item = target
                    else:
                        effective = mutation.effective_from
                        if (target.fact_key is None or effective is None
                                or target.valid_from is not None and effective <= target.valid_from
                                or target.valid_to is not None and effective >= target.valid_to):
                            raise ValueError("事实修订的生效时间必须位于旧事实有效区间内")
                        if mutation.content == target.content:
                            continue
                        item = MemoryRecord(
                            id=f"memory_{uuid4().hex}", user_id=self.user_id, content=mutation.content,
                            memory_type=target.memory_type, importance=target.importance,
                            created_at=now, updated_at=now, metadata=deepcopy(target.metadata),
                            layer=target.layer, task_id=target.task_id, fact_key=target.fact_key,
                            valid_from=effective, valid_to=target.valid_to, supersedes_id=target.id,
                        )
                        target.valid_to = effective
                        target.updated_at = now
                        target.version += 1
                        upserts.append(target)
                        snapshot.records.append(item)
                evidence_refs = []
                for evidence_id in dict.fromkeys(mutation.evidence_ids):
                    if evidence_id in evidence_versions:
                        evidence_refs.append({"memory_id": evidence_id, "version": evidence_versions[evidence_id]})
                    else:
                        message = mutation.evidence_messages[evidence_id]
                        evidence_refs.append({
                            "message_id": evidence_id, "role": message.role, "content": message.content,
                            "timestamp": message.timestamp.isoformat(),
                        })
                item.metadata["write_evidence"] = evidence_refs
                upserts.append(item)
                returned_ids.append(item.id)

            if not upserts:
                return []
            versions = {record.id: record.version for record in upserts}
            snapshot.index_events.extend(
                MemoryIndexEvent(f"event_{uuid4().hex}", self.user_id, item.id, "upsert", item.version, now)
                for item in upserts
            )
            for summary in snapshot.records:
                if summary.layer != "summary" or summary.summary_state == "stale":
                    continue
                if any(ref.memory_id in versions and versions[ref.memory_id] != ref.version
                       for ref in summary.source_refs):
                    summary.summary_state = "stale"
                    summary.updated_at = now
                    summary.version += 1
                    snapshot.index_events.append(MemoryIndexEvent(
                        f"event_{uuid4().hex}", self.user_id, summary.id, "delete", summary.version, now,
                    ))
            try:
                saved = self.repository.commit(snapshot)
            except MemoryConflict:
                if attempt == 2:
                    raise
                continue
            saved_by_id = {record.id: record for record in saved.records}
            return [saved_by_id[memory_id] for memory_id in returned_ids]
