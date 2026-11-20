"""业务记忆的错误，不吞掉持久化故障，也不把 TODO 伪装为空结果。"""


class MemoryConflict(RuntimeError):
    """提交的业务快照或记录版本已经过时，应重新读取后决定如何处理。"""


class MemoryIndexPending(RuntimeError):
    """业务记录尚有待同步事件；不能把旧索引伪装成最新业务内容。"""


class MemoryEmbeddingNotConfigured(RuntimeError):
    """明确选择向量/混合模式，却没有注入共享嵌入组件。"""


class MemorySummaryStale(RuntimeError):
    """摘要依赖的原始记录已变化或被删除，需要重新生成。"""
