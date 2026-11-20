"""M04：BM25 基线、向量召回和按名次融合；有效性过滤在调用本模块前完成。"""

from ..retrieval import bm25_rank, fuse_rankings, tokenize

from .config import MemoryConfig
from .errors import MemoryEmbeddingNotConfigured
from .ports import MemoryVectorIndex
from .types import MemoryHit, MemoryRecord, SearchMode


class LexicalMemoryRetriever:
    def __init__(self, config: MemoryConfig | None = None):
        self.config = config if config is not None else MemoryConfig()

    def tokenize(self, text: str) -> list[str]:
        """切分中英文检索词项，保留函数名与错误码。

        ASCII 字母、数字与下划线的连续段转为小写。常用汉字按相邻两字
        切分，单字保留。词项按原顺序返回，保留重复出现的词项。
        """
        return tokenize(text)

    def search(self, query: str, candidates: list[MemoryRecord], limit: int) -> list[MemoryHit]:
        """在已筛选的候选中计算 BM25，按分数降序和记忆 ID 升序返回命中。"""
        ranked = bm25_rank(
            query, candidates, limit, key=lambda item: item.id, text=lambda item: item.content,
            k1=self.config.bm25_k1, b=self.config.bm25_b, split=self.tokenize,
        )
        return [MemoryHit(item, score=score, lexical_score=score) for item, score in ranked]


class HybridMemoryRetriever:

    def __init__(
        self, lexical: LexicalMemoryRetriever,
        vector: MemoryVectorIndex | None = None, config: MemoryConfig | None = None,
    ):
        self.lexical = lexical
        self.vector = vector
        self.config = config if config is not None else MemoryConfig()

    def fuse(
        self, lexical_hits: list[MemoryHit], vector_hits: list[MemoryHit], limit: int,
    ) -> list[MemoryHit]:
        """按记忆 ID 融合两路加权排名，保留首次命中的名次和原始分数。"""
        routes = [
            ("lexical", self.config.lexical_weight, [(hit.memory, hit.score) for hit in lexical_hits]),
            ("vector", self.config.vector_weight, [(hit.memory, hit.score) for hit in vector_hits]),
        ]
        ranked = fuse_rankings(routes, limit, key=lambda item: item.id, rrf_k=self.config.rrf_k)
        return [MemoryHit(
            hit.item, score=hit.score, lexical_score=hit.scores.get("lexical"),
            vector_score=hit.scores.get("vector"), ranks=hit.ranks,
        ) for hit in ranked]

    def search(
        self, query: str, candidates: list[MemoryRecord], limit: int, mode: SearchMode,
    ) -> list[MemoryHit]:
        """按指定模式检索已筛选的候选，返回最多 limit 条命中。

        混合模式取得两路排名并在融合后截取结果。向量检索需要已配置的
        适配器，适配器的检索异常会传给调用方。
        """
        if mode not in ("lexical", "vector", "hybrid"):
            raise ValueError("不支持的记忆检索模式")
        if type(limit) is not int or limit < 0:
            raise ValueError("limit 必须是非负整数")
        if not candidates or limit == 0:
            return []
        if mode == "lexical":
            return self.lexical.search(query, candidates, limit)

        vector = self.vector
        if vector is None:
            raise MemoryEmbeddingNotConfigured("向量检索需要配置记忆向量适配器")
        if mode == "vector":
            return vector.search(query, candidates, limit)

        source_limit = len(candidates)
        lexical_hits = self.lexical.search(query, candidates, source_limit)
        vector_hits = vector.search(query, candidates, source_limit)
        return self.fuse(lexical_hits, vector_hits, limit)
