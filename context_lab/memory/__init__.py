"""唯一的分层、时间有效性和混合检索记忆框架，不自动联网。

EmbeddingMemoryBridge 需显式从 embedding_bridge 导入，以保留纯本地模式。
"""

from .codec import MemoryCodec
from .config import MemoryConfig
from .errors import MemoryConflict, MemoryEmbeddingNotConfigured, MemoryIndexPending, MemorySummaryStale
from .hierarchy import MemoryHierarchy
from .maintenance import MemoryCleanupWorker
from .policy import MemoryPolicy
from .repository import JsonMemoryRepository
from .retention import MemoryRetentionPolicy
from .retrieval import HybridMemoryRetriever, LexicalMemoryRetriever
from .service import MemoryService
from .types import (
    MemoryCleanupPlan, MemoryCleanupReport, MemoryContext, MemoryHit, MemoryIndexEvent, MemoryMutation, MemoryQuery,
    MemoryRecord, MemorySnapshot, MemorySourceRef, MemorySyncReport,
)
from .writer import MemoryWriter

__all__ = [
    "MemoryCodec", "MemoryConfig", "MemoryConflict", "MemoryEmbeddingNotConfigured",
    "MemoryIndexPending", "MemorySummaryStale", "MemoryHierarchy", "MemoryPolicy",
    "JsonMemoryRepository", "HybridMemoryRetriever", "LexicalMemoryRetriever", "MemoryService",
    "MemoryContext", "MemoryHit", "MemoryIndexEvent", "MemoryMutation", "MemoryQuery",
    "MemoryRecord", "MemorySnapshot", "MemorySourceRef", "MemorySyncReport", "MemoryWriter",
    "MemoryRetentionPolicy", "MemoryCleanupPlan", "MemoryCleanupReport", "MemoryCleanupWorker",
]
