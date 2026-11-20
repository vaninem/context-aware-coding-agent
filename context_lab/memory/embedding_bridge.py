"""M05：显式导入此模块才需要 embedding 可选依赖；不重写已完成的向量底座。"""

import hashlib
from urllib.parse import quote

from ..embeddings import EmbeddingChunk, IncrementalEmbeddingIndex
from ..embeddings.errors import EmbeddingRequestError, IndexNotReady
from .config import MemoryConfig
from .errors import MemoryIndexPending
from .types import MemoryHit, MemoryRecord


class EmbeddingMemoryBridge:
    def __init__(
        self, index: IncrementalEmbeddingIndex, user_id: str, config: MemoryConfig | None = None,
        *, scope: str | None = None,
    ):
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id 必须为非空字符串")
        self.index = index
        self.user_id = user_id
        self.namespace = "memory:" + quote(user_id, safe="")
        if scope is not None:
            self.namespace += ":" + quote(scope, safe="")
        self.config = config if config is not None else MemoryConfig()

    def apply(self, memory_id: str, record: MemoryRecord | None) -> None:
        """将当前用户的记忆同步为正文块，或清空指定来源的块。

        record 为 None 或失效摘要时清空来源。正常记录的正文使用固定的
        body 块 ID，索引元数据以业务记录为准。向量生成由 process_pending 负责。
        """
        if not isinstance(memory_id, str) or not memory_id.strip():
            raise ValueError("memory_id 必须为非空字符串")
        if record is not None:
            if not isinstance(record, MemoryRecord):
                raise ValueError("record 必须为 MemoryRecord 或 None")
            if record.id != memory_id or record.user_id != self.user_id:
                raise ValueError("记忆 ID 或用户与索引来源不一致")

        if record is None or (record.layer == "summary" and record.summary_state == "stale"):
            chunks = []
        else:
            if not isinstance(record.metadata, dict):
                raise ValueError("记忆 metadata 必须为字典")
            metadata = {
                **record.metadata,
                "user_id": record.user_id,
                "memory_version": record.version,
                "layer": record.layer,
                "task_id": record.task_id,
                "valid_from": record.valid_from.isoformat() if record.valid_from else None,
                "valid_to": record.valid_to.isoformat() if record.valid_to else None,
                "summary_state": record.summary_state,
            }

            chunk = EmbeddingChunk(id="body", text=record.content, metadata=metadata)
            chunks = [chunk]
        self.index.sync_source(self.namespace, memory_id, chunks)

    def search(self, query: str, candidates: list[MemoryRecord], limit: int) -> list[MemoryHit]:
        """在当前用户的业务候选中按向量相似度返回至多 limit 条命中。

        查询只编码一次。每轮从数据库排名开头筛选，结果不足时扩大窗口。
        命中须匹配候选的正文块、业务版本和正文哈希。索引有未完成或失败
        的块时抛出 IndexNotReady，候选索引缺失或过时时抛出 MemoryIndexPending。
        """
        if type(limit) is not int or limit < 0:
            raise ValueError("limit 必须是非负整数")
        if limit == 0 or not candidates:
            return []
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query 必须是非空字符串")
        if any(not isinstance(record, MemoryRecord) or record.user_id != self.user_id
               for record in candidates):
            raise ValueError("候选必须是当前用户的 MemoryRecord")

        record_by_id = {record.id: record for record in candidates}
        target = min(limit, len(record_by_id))
        status = self.index.status(self.namespace)
        if status.pending or status.running or status.failed:
            raise IndexNotReady(
                f"记忆索引尚未完整就绪（pending={status.pending}, running={status.running}, failed={status.failed}）"
            )
        if status.ready < target:
            raise MemoryIndexPending("记忆索引的就绪块数少于查询所需的候选数")

        vectors = self.index.client.embed([query])
        if not isinstance(vectors, list) or len(vectors) != 1:
            raise EmbeddingRequestError("查询编码必须返回一个向量")
        query_vector = vectors[0]
        window = min(status.ready, target * self.config.vector_fetch_multiplier)
        while True:
            results = []
            hits = self.index.store.search_ready(
                self.index.profile_id, self.namespace, query_vector, limit=window,
            )
            for hit in hits:
                chunk = hit.chunk
                record = record_by_id.get(chunk.source_id)
                if record is None:
                    continue
                version = chunk.metadata.get("memory_version")
                content_hash = hashlib.sha256(record.content.encode("utf-8")).hexdigest()
                if (chunk.chunk_id != "body" or type(version) is not int or version != record.version
                        or chunk.content_hash != content_hash):
                    raise MemoryIndexPending(f"记忆 {record.id} 的正文索引与业务记录不一致")

                results.append(MemoryHit(memory=record, score=hit.score, vector_score=hit.score))
                if len(results) == target:
                    return results
            if window == status.ready:
                raise MemoryIndexPending("扫描全部就绪块后仍缺少候选记忆的正文索引")
            window = min(status.ready, window * 2)

    def process_pending(self, limit: int | None = None) -> int:
        """已提供的薄委托；只有用户显式调用时才执行已有向量生成流程。"""
        return self.index.process_pending(limit=limit)
