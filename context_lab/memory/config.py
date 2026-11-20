"""只保存配置；默认词法检索，不自动读取 .env 或联网。"""

from dataclasses import dataclass
import math
from numbers import Real

from .types import SearchMode


@dataclass(frozen=True)
class MemoryConfig:
    search_mode: SearchMode = "lexical"
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    rrf_k: int = 60
    lexical_weight: float = 1.0
    vector_weight: float = 1.0
    vector_fetch_multiplier: int = 4
    context_tokens: int = 2000
    working_ratio: float = 0.5
    summary_ratio: float = 0.25
    facts_ratio: float = 0.2
    archive_ratio: float = 0.05
    history_retention_days: int = 90  # archive fact 与普通经历共用；起算时间由策略区分
    auto_cleanup_history: bool = False
    history_cleanup_interval_seconds: int = 3600
    history_cleanup_batch_size: int = 100

    def __post_init__(self):
        if self.search_mode not in ("lexical", "vector", "hybrid"):
            raise ValueError("不支持的记忆检索模式")
        numeric = (self.bm25_k1, self.bm25_b, self.lexical_weight, self.vector_weight,
                   self.working_ratio, self.summary_ratio, self.facts_ratio, self.archive_ratio)
        if any(isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) for value in numeric):
            raise ValueError("记忆配置的分数与比例必须是有限数值")
        if self.bm25_k1 <= 0 or not 0 <= self.bm25_b <= 1:
            raise ValueError("BM25 参数无效")
        for name in ("rrf_k", "vector_fetch_multiplier", "context_tokens", "history_retention_days",
                     "history_cleanup_interval_seconds", "history_cleanup_batch_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} 必须是正整数")
        if type(self.auto_cleanup_history) is not bool:
            raise ValueError("auto_cleanup_history 必须是 bool")
        if self.lexical_weight < 0 or self.vector_weight < 0 or self.lexical_weight + self.vector_weight <= 0:
            raise ValueError("融合权重必须非负且不能同时为零")
        ratios = (self.working_ratio, self.summary_ratio, self.facts_ratio, self.archive_ratio)
        if any(value < 0 for value in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
            raise ValueError("四类上下文预算比例必须非负且合计为 1")
