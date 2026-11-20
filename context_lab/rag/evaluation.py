"""检索质量和索引性能的离线评测。"""

from dataclasses import dataclass
from math import ceil
from time import perf_counter, sleep
from typing import Callable

from ..retrieval import fuse_rankings
from ..types import SearchHit
from .rerank import RerankRequestError
from .types import RetrievalCase, RetrievalMetrics, VectorIndexComparison


def fuse_benchmark_hits(lexical: list[SearchHit], dense: list[SearchHit], limit: int) -> list[SearchHit]:
    """按块 ID 对词法和向量两路排名做等权 RRF。"""
    routes = [(name, 1.0, [(hit.chunk, hit.score) for hit in hits])
              for name, hits in (("lexical", lexical), ("vector", dense))]
    return [SearchHit(hit.item, hit.score, ranks=hit.ranks, fusion_score=hit.score)
            for hit in fuse_rankings(routes, limit, key=lambda chunk: chunk.id)]


def rerank_benchmark_hits(reranker, query: str, hits: list[SearchHit], k: int) -> list[SearchHit]:
    """重排候选，遇到暂时的接口错误时最多重试两次。"""
    if not hits:
        return []
    for attempt in range(3):
        try:
            return reranker.rerank(query, hits, k)
        except RerankRequestError as exc:
            if not exc.retryable or attempt == 2:
                raise
            sleep(2 ** attempt)


@dataclass(frozen=True)
class RerankComparison:
    name: str
    candidates: int
    reranker: object


def compare_rerankers(
    query: str, hits: list[SearchHit], retrieval_ms: float, comparisons: tuple[RerankComparison, ...],
    *, source: str = "vector",
) -> dict[str, tuple[list[SearchHit], float]]:
    """重排同一候选池，同时记录该候选池的原序。"""
    rankings = {}
    for comparison in comparisons:
        pool = hits[:comparison.candidates]
        rankings.setdefault(f"{source}_top{comparison.candidates}", (pool, retrieval_ms))
        begun = perf_counter()
        ranked = rerank_benchmark_hits(comparison.reranker, query, pool, len(pool))
        rankings[comparison.name] = (ranked, retrieval_ms + (perf_counter() - begun) * 1000)
    return rankings


def evaluate_retrieval(
    cases: list[RetrievalCase], search: Callable[[str, int], list[SearchHit]], limit: int,
) -> RetrievalMetrics:
    """按块 ID 计算每题的 Recall@K 和 RR@K，再对各题取平均。

    P95 只计搜索回调耗时，取排序后第 ceil(0.95 * 题数) 个值。
    无正确标签的题按零分计。空题集返回全零指标。
    """
    if type(limit) is not int or limit <= 0:
        raise ValueError("limit 必须是正整数")
    case_count = len(cases)
    if case_count == 0:
        return RetrievalMetrics(0, 0.0, 0.0, 0.0)

    latencies_ms = []
    recall_sum = 0.0
    rr_sum = 0.0
    for case in cases:
        start = perf_counter()
        hits = search(case.query, limit)
        elapsed_ms = (perf_counter() - start) * 1000
        latencies_ms.append(elapsed_ms)
        hit_ids = [hit.chunk.id for hit in hits[:limit]]
        expected = case.expected_ids
        matched = set(hit_ids) & expected
        recall = len(matched) / len(expected) if expected else 0.0
        rr = 0.0
        for rank, hit_id in enumerate(hit_ids, start=1):
            if hit_id in expected:
                rr = 1 / rank
                break
        recall_sum += recall
        rr_sum += rr
    ordered = sorted(latencies_ms)
    p95_ms = ordered[ceil(0.95 * case_count) - 1]
    return RetrievalMetrics(
        cases=case_count,
        recall_at_k=recall_sum / case_count,
        mrr_at_k=rr_sum / case_count,
        p95_ms=p95_ms,
    )


def compare_vector_indexes(
    queries: list[str], exact: Callable[[str, int], list[SearchHit]],
    approximate: Callable[[str, int], list[SearchHit]], limit: int,
) -> VectorIndexComparison:
    """对同一批查询比较精确与近似检索的 Top-K 重合率及 P95 延迟。

    两个回调应使用同一数据与过滤条件。每题以精确结果的唯一块 ID 数为分母，
    精确结果为空的题计零分。空查询列表返回全零指标。
    """
    if type(limit) is not int or limit <= 0:
        raise ValueError("limit 必须是正整数")
    query_count = len(queries)
    if query_count == 0:
        return VectorIndexComparison(0, 0.0, 0.0, 0.0)

    exact_ms = []
    approximate_ms = []
    overlap_sum = 0.0
    for query in queries:
        exact_start = perf_counter()
        exact_hits = exact(query, limit)
        exact_ms.append((perf_counter() - exact_start) * 1000)
        approximate_start = perf_counter()
        approximate_hits = approximate(query, limit)
        approximate_ms.append((perf_counter() - approximate_start) * 1000)
        exact_ids = {hit.chunk.id for hit in exact_hits[:limit]}
        approximate_ids = {hit.chunk.id for hit in approximate_hits[:limit]}
        if exact_ids:
            overlap_sum += len(exact_ids & approximate_ids) / len(exact_ids)

    p95_index = ceil(0.95 * query_count) - 1
    return VectorIndexComparison(
        queries=query_count,
        mean_overlap_at_k=overlap_sum / query_count,
        exact_p95_ms=sorted(exact_ms)[p95_index],
        approximate_p95_ms=sorted(approximate_ms)[p95_index],
    )
