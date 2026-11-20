"""用 RepoQA Python 的原文与目标函数标注评测本地 RAG。"""

import gzip
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from math import ceil
from pathlib import Path
from tempfile import TemporaryDirectory
from time import sleep
from time import perf_counter
from typing import Callable

from ..models import ModelRequestError
from ..storage.json_store import JsonFileStore
from ..tools.rag import RAGTool
from ..types import SearchHit
from .evaluation import RerankComparison, compare_rerankers, fuse_benchmark_hits, rerank_benchmark_hits
from .query import plan_query
from .source_coverage import covered_lines, target_lines
from .types import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_TOKENS, QueryPlan


def evaluate_repoqa(dataset: Path, k: int = 5, *, vectors=None, reranker=None, candidates: int = 50,
                    rerank_source: str = "hybrid", repo_limit: int | None = None,
                    rerank_candidates: int | None = None, query_model=None,
                    hyde: bool = False, multi_query: bool = False, query_workers: int = 1,
                    query_cache: Path | None = None, source_coverage: bool = False,
                    comparisons: tuple[RerankComparison, ...] = (), rerank_workers: int = 1,
                    comparison_source: str = "vector",
                    progress: Callable[[str], None] | None = None) -> dict:
    """按目标函数的路径和原文区间判定，查询只使用自然语言描述。"""
    if type(k) is not int or k <= 0:
        raise ValueError("k 必须为正整数")
    if (vectors is not None or reranker is not None) and candidates < k:
        raise ValueError("候选数必须大于等于 k")
    if repo_limit is not None and repo_limit <= 0:
        raise ValueError("仓库数量必须为正整数")
    enhanced = hyde or multi_query
    if enhanced and (vectors is None or query_model is None):
        raise ValueError("HyDE 和多查询需要配置向量检索和查询增强模型")
    if query_workers <= 0:
        raise ValueError("查询生成的并行数必须为正整数")
    if rerank_workers <= 0 or (comparisons and (vectors is None or any(
        not k <= variant.candidates <= candidates for variant in comparisons
    ))):
        raise ValueError("重排对比需要向量、足够的候选及正数并行数")
    if comparison_source not in {"vector", "hybrid"}:
        raise ValueError("重排对比候选来源必须是 vector 或 hybrid")
    if rerank_candidates is not None and not k <= rerank_candidates <= candidates:
        raise ValueError("重排候选数必须在 k 和召回候选数之间")
    if rerank_source not in {"hybrid", "vector", "enhanced"}:
        raise ValueError("重排候选来源必须是 hybrid、vector 或 enhanced")
    if rerank_source != "hybrid" and (vectors is None or reranker is None):
        raise ValueError("向量重排需要同时配置向量检索和重排")
    if rerank_source == "enhanced" and not enhanced:
        raise ValueError("增强重排需要启用 HyDE 或多查询")
    encoding = None
    if source_coverage:
        import tiktoken

        encoding = tiktoken.get_encoding("o200k_base")
        if candidates < max(k, 10):
            raise ValueError("源码覆盖评测的候选数须覆盖前 10 个结果")
    opener = gzip.open if dataset.suffix == ".gz" else open
    with opener(dataset, "rt", encoding="utf-8") as stream:
        repositories = json.load(stream)["python"]
    rows: dict[str, list[dict]] = {"bm25": []}
    if vectors is not None:
        rows.update(vector=[], hybrid=[])
    if hyde:
        rows["vector_hyde"] = []
    if multi_query:
        rows["vector_multi_query"] = []
    if hyde and multi_query:
        rows["vector_hyde_multi_query"] = []
    candidate_source = rerank_source if vectors is not None else "bm25"
    if rerank_source == "enhanced":
        candidate_source = next(reversed(rows))
    reranked_name = f"{candidate_source}_rerank"
    if reranker is not None:
        rows[reranked_name] = []
    for variant in comparisons:
        rows.setdefault(f"{comparison_source}_top{variant.candidates}", [])
        rows[variant.name] = []
    indexing_ms = 0.0
    plan_store = JsonFileStore(query_cache) if query_cache is not None and enhanced else None
    cached_plans = plan_store.read(default={}) if plan_store is not None else {}
    cache_prefix = ["repoqa-query-v2", getattr(query_model, "base_url", None), getattr(query_model, "model_name", None), hyde, multi_query]
    with TemporaryDirectory(prefix="context-lab-repoqa-") as folder:
        for index, repository in enumerate(repositories[:repo_limit]):
            if not repository.get("needles"):
                continue
            workspace = Path(folder) / str(index)
            rag = RAGTool(workspace)
            content = repository["content"]
            for source, text in sorted(content.items()):
                relative = Path(source)
                if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                    raise ValueError(f"RepoQA 文件路径不合法：{source}")
                if relative.suffix.lower() != ".py":
                    continue
                path = workspace / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(text.encode("utf-8"))
                rag.add_document(relative)
            if progress:
                progress(f"{repository['repo']} · {len(rag.chunks)} 个代码块")
            matrix = None
            if vectors is not None and rag.chunks:
                begun = perf_counter()
                matrix = vectors.prepare(rag.chunks)
                indexing_ms += (perf_counter() - begun) * 1000
            needles = repository["needles"]
            if enhanced:
                def generate_plan(needle):
                    query = needle["description"]
                    key = hashlib.sha256(json.dumps([*cache_prefix, query]).encode()).hexdigest()
                    if key in cached_plans:
                        lexical_queries, vector_queries, generation_ms = cached_plans[key]
                        return key, QueryPlan(tuple(lexical_queries), tuple(vector_queries)), generation_ms
                    begun = perf_counter()
                    for attempt in range(3):
                        try:
                            plan = plan_query(query, query_model, rewrite=multi_query, hyde=hyde)
                            return key, plan, (perf_counter() - begun) * 1000
                        except ValueError as exc:
                            if attempt == 2:
                                raise ValueError(f"{repository['repo']} 的查询增强格式错误：{exc}") from None
                        except ModelRequestError as exc:
                            if not exc.retryable or attempt == 2:
                                raise ModelRequestError(f"{repository['repo']} 的查询增强失败：{exc}") from None
                            sleep(2 ** attempt)

                with ThreadPoolExecutor(max_workers=query_workers) as executor:
                    plans = []
                    for key, plan, generation_ms in executor.map(generate_plan, needles):
                        plans.append((plan, generation_ms))
                        if plan_store is not None and key not in cached_plans:
                            cached_plans[key] = [plan.lexical_queries, plan.vector_queries, generation_ms]
                            plan_store.write(cached_plans)
            else:
                plans = [(None, 0.0)] * len(needles)

            def evaluate_needle(item):
                needle, (plan, generation_ms) = item
                query = needle["description"]
                source = Path(needle["path"]).as_posix()
                raw = content[needle["path"]].encode("utf-8")
                byte_start, byte_end = needle["start_byte"], needle["end_byte"]
                if not 0 <= byte_start < byte_end <= len(raw):
                    raise ValueError(f"RepoQA 函数偏移不合法：{source}")
                start = len(raw[:byte_start].decode("utf-8"))
                end = start + len(raw[byte_start:byte_end].decode("utf-8"))
                begun = perf_counter()
                lexical = rag.search(query, candidates if len(rows) > 1 or source_coverage else k)
                lexical_ms = (perf_counter() - begun) * 1000
                rankings = {"bm25": (lexical, lexical_ms)}
                if vectors is not None:
                    begun = perf_counter()
                    if plan is not None and matrix is not None:
                        original = vectors.score_queries([query], matrix)[0]
                        dense = vectors.rank(rag.chunks, original, candidates)
                    else:
                        dense = vectors.search(query, rag.chunks, matrix, candidates) if matrix is not None else []
                    dense_ms = (perf_counter() - begun) * 1000
                    rankings["vector"] = (dense, dense_ms)
                    begun = perf_counter()
                    hybrid = fuse_benchmark_hits(lexical, dense, candidates)
                    hybrid_ms = lexical_ms + dense_ms + (perf_counter() - begun) * 1000
                    rankings["hybrid"] = (hybrid, hybrid_ms)
                    if plan is not None:
                        if matrix is None:
                            rankings.update({name: ([], generation_ms) for name in rows if name.startswith("vector_") and name != reranked_name})
                        else:
                            rankings.update(_enhanced_rankings(vectors, rag.chunks, matrix, plan, original,
                                                               candidates, dense_ms + generation_ms, hyde, multi_query))
                if reranker is not None:
                    rerank_hits, rerank_ms = rankings[candidate_source]
                    if rerank_candidates is not None:
                        rerank_hits = rerank_hits[:rerank_candidates]
                    begun = perf_counter()
                    reranked = rerank_benchmark_hits(reranker, query, rerank_hits, len(rerank_hits) if source_coverage else k)
                    rankings[reranked_name] = (
                        reranked, rerank_ms + (perf_counter() - begun) * 1000,
                    )
                if comparisons:
                    pool, pool_ms = rankings[comparison_source]
                    rankings.update(compare_rerankers(query, pool, pool_ms, comparisons, source=comparison_source))
                indexed_target = any(chunk.source == source and chunk.start < end and start < chunk.end for chunk in rag.chunks)
                lines = target_lines(content[source], start, end) if source_coverage else []

                def first_match(hits: list[SearchHit]) -> int | None:
                    return next((rank for rank, hit in enumerate(hits, 1)
                                 if hit.chunk.source == source and hit.chunk.start < end and start < hit.chunk.end), None)

                result = {}
                for name, (hits, latency) in rankings.items():
                    rank = first_match(hits)
                    candidate_rank = first_match(rerank_hits) if reranker is not None and name == reranked_name else rank
                    row = {"repo": repository["repo"], "target": needle["name"], "path": source,
                                       "rank": rank if rank is not None and rank <= k else None,
                                       "candidate_rank": candidate_rank, "target_indexed": indexed_target, "latency_ms": latency}
                    if encoding is not None:
                        files = list(dict.fromkeys(hit.chunk.source for hit in hits))
                        row.update(
                            file_rank=files.index(source) + 1 if source in files else None,
                            target_lines=len(lines),
                            covered_lines_at_2000=covered_lines(hits, content, source, lines, 2000, encoding),
                            covered_lines_at_8000=covered_lines(hits, content, source, lines, 8000, encoding),
                            returned_chunks=len(hits),
                        )
                    if plan is not None and name.startswith("vector_"):
                        row.update(rewrites=list(plan.lexical_queries[1:]), hypothetical_document=next(iter(plan.vector_queries[1:]), None),
                                   generation_ms=generation_ms)
                    result[name] = row
                return result

            with (ThreadPoolExecutor(rerank_workers) if rerank_workers > 1 else nullcontext()) as pool:
                inputs = zip(needles, plans, strict=True)
                results = pool.map(evaluate_needle, inputs) if pool is not None else map(evaluate_needle, inputs)
                for result in results:
                    for name, row in result.items():
                        rows[name].append(row)
            if comparisons and progress:
                progress(f"RepoQA · 已评测 {len(rows['vector'])} 题")
            if enhanced and progress:
                progress(f"{repository['repo']} · 已完成 {len(rows['vector'])} 题")
    if not rows["bm25"]:
        raise ValueError("数据里没有 RepoQA Python 标注样本")
    digest = hashlib.sha256()
    with dataset.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            digest.update(block)
    common = {
        "dataset": dataset.name, "dataset_sha256": digest.hexdigest(),
        "language": "python", "k": k, "chunk_size": DEFAULT_CHUNK_TOKENS,
        "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
    }
    if source_coverage:
        common["source_coverage"] = {"tokenizer": encoding.name, "budgets": [2000, 8000],
                                     "candidate_limit": candidates, "deduplicate": "source_spans"}
    metrics = {name: _metrics(results, k) for name, results in rows.items()}
    if len(rows) == 1:
        return {**common, "retriever": "bm25", **metrics["bm25"]}
    return {**common, "candidate_limit": candidates, "vector_index_ms": indexing_ms,
            "embedding_model": vectors.client.config.model if vectors is not None else None,
            "rerank_model": reranker.config.model if reranker is not None else None,
            "rerank_candidates": (rerank_candidates or candidates) if reranker is not None else None,
            "query_model": getattr(query_model, "model_name", None) if enhanced else None,
            "rerank_source": candidate_source if reranker is not None else None, "runs": metrics}


def _enhanced_rankings(vectors, chunks, matrix, plan, original, candidates, base_ms, hyde, multi_query):
    """保留原查询的半数分数贡献，多条改写先取均值。"""
    results, families = {}, []
    total_score_ms = 0.0
    for name, enabled, queries in (("vector_hyde", hyde, plan.vector_queries[1:]),
                                   ("vector_multi_query", multi_query, plan.lexical_queries[1:])):
        if not enabled:
            continue
        begun = perf_counter()
        scores = vectors.score_queries(list(queries), matrix) if queries else []
        if scores:
            signal = sum(scores) / len(scores)
            families.append(signal)
        else:
            signal = original
        score_ms = (perf_counter() - begun) * 1000
        total_score_ms += score_ms
        begun = perf_counter()
        hits = vectors.rank(chunks, (original + signal) / 2, candidates)
        results[name] = (hits, base_ms + score_ms + (perf_counter() - begun) * 1000)
    if hyde and multi_query:
        begun = perf_counter()
        signal = sum(families) / len(families) if families else original
        hits = vectors.rank(chunks, (original + signal) / 2, candidates)
        results["vector_hyde_multi_query"] = (hits, base_ms + total_score_ms + (perf_counter() - begun) * 1000)
    return results


def _metrics(rows: list[dict], k: int) -> dict:
    count = len(rows)
    latencies = sorted(row["latency_ms"] for row in rows)
    hit_rate = sum(row["rank"] is not None for row in rows) / count
    metrics = {
        "cases": count,
        "repositories": len({row["repo"] for row in rows}),
        "hit_at_k": hit_rate, "recall_at_k": hit_rate,
        "candidate_hit_rate": sum(row["candidate_rank"] is not None for row in rows) / count,
        "mrr_at_k": sum(1 / row["rank"] for row in rows if row["rank"] is not None) / count,
        "hit_by_rank": {str(rank): sum(row["rank"] is not None and row["rank"] <= rank for row in rows) / count
                        for rank in range(1, k + 1)},
        "p95_ms": latencies[ceil(0.95 * count) - 1],
        "target_index_coverage": sum(row["target_indexed"] for row in rows) / count,
        "details": rows,
    }
    if "file_rank" in rows[0]:
        metrics.update({f"file_hit_at_{cutoff}": sum(row["file_rank"] is not None and row["file_rank"] <= cutoff
                                                    for row in rows) / count for cutoff in (1, 5, 10)})
        metrics["file_mrr_at_10"] = sum(1 / row["file_rank"] for row in rows
                                        if row["file_rank"] is not None and row["file_rank"] <= 10) / count
        for budget in (2000, 8000):
            covered = f"covered_lines_at_{budget}"
            metrics[f"half_function_at_{budget}"] = sum(2 * row[covered] >= row["target_lines"] for row in rows) / count
            metrics[f"full_function_at_{budget}"] = sum(row[covered] == row["target_lines"] for row in rows) / count
    return metrics
