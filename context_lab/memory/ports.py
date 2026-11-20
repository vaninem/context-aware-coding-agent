"""只定义边界，不导入 psycopg/httpx，不创建向量客户端。"""

from typing import Protocol

from .types import MemoryHit, MemoryRecord


class MemoryVectorIndex(Protocol):
    user_id: str
    namespace: str

    def apply(self, memory_id: str, record: MemoryRecord | None) -> None: ...

    def search(self, query: str, candidates: list[MemoryRecord], limit: int) -> list[MemoryHit]: ...

    def process_pending(self, limit: int | None = None) -> int: ...
