"""用 BEIR SciFact 的官方测试集评测分块后的文档检索。"""

import csv
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from functools import lru_cache
from io import TextIOWrapper
from math import ceil, log2
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Callable
from zipfile import BadZipFile, ZipFile

from ..retrieval import bm25_rank, tokenize
from ..tools.rag import RAGTool
from ..types import SearchHit
from .evaluation import RerankComparison, compare_rerankers, fuse_benchmark_hits, rerank_benchmark_hits
from .recall import index_text


def _read_scifact(dataset: Path):
    with ZipFile(dataset) as archive:
        with archive.open("scifact/corpus.jsonl") as stream:
            corpus = {row["_id"]: (row["title"], row["text"]) for row in map(json.loads, stream)}
        with archive.open("scifact/queries.jsonl") as stream:
            queries = {row["_id"]: row["text"] for row in map(json.loads, stream)}
        with TextIOWrapper(archive.open("scifact/qrels/test.tsv"), encoding="utf-8") as stream:
            qrels: dict[str, dict[str, int]] = defaultdict(dict)
            for row in csv.DictReader(stream, delimiter="\t"):
                qrels[row["query-id"]][row["corpus-id"]] = int(row["score"])
    return corpus, queries, qrels


def _score_documents(hits: list[SearchHit], labels: dict[str, int], k: int) -> dict:
    docs = list(dict.fromkeys(Path(hit.chunk.source).stem for hit in hits))[:k]
    grades = [labels.get(doc_id, 0) for doc_id in docs]
    ideal = sorted(labels.values(), reverse=True)[:k]
    dcg = sum(grade / log2(rank + 1) for rank, grade in enumerate(grades, 1))
    idcg = sum(grade / log2(rank + 1) for rank, grade in enumerate(ideal, 1))
    first = next((rank for rank, grade in enumerate(grades, 1) if grade > 0), None)
    return {"top_document_ids": docs, "first_relevant_rank": first, "hit": first is not None,
            "recall": sum(grade > 0 for grade in grades) / sum(grade > 0 for grade in labels.values()),
            "rr": 1 / first if first else 0.0, "ndcg": dcg / idcg}


def evaluate_scifact(dataset: Path, k: int, *, vectors=None, reranker=None, candidates: int = 50,
                     rerank_candidates: int | None = None, progress: Callable[[str], None] | None = None,
                     comparisons: tuple[RerankComparison, ...] = (), rerank_workers: int = 1,
                     comparison_source: str = "vector", query_ids: frozenset[str] | None = None) -> dict:
    """检索整个语料的文档块，按官方相关文档 ID 去重和计算排名指标。"""
    rerank_candidates = rerank_candidates if rerank_candidates is not None else min(candidates, max(k, 20))
    if not 1 <= k <= rerank_candidates <= candidates:
        raise ValueError("需要满足 1 <= k <= 重排候选数 <= 召回候选数")
    if rerank_workers <= 0 or (comparisons and (vectors is None or any(
        not k <= variant.candidates <= candidates for variant in comparisons
    ))):
        raise ValueError("重排对比需要向量、足够的候选及正数并行数")
    if comparison_source not in {"vector", "hybrid"}:
        raise ValueError("重排对比候选来源必须是 vector 或 hybrid")
    try:
        corpus, queries, qrels = _read_scifact(dataset)
    except (BadZipFile, KeyError) as exc:
        raise ValueError(f"SciFact 数据包格式错误：{exc}") from exc
    if not qrels or any(query not in queries or not any(grade > 0 for grade in labels.values())
                        or labels.keys() - corpus.keys() for query, labels in qrels.items()):
        raise ValueError("SciFact 测试标注缺少有效查询或相关文档")
    if query_ids is not None:
        if not query_ids or query_ids - qrels.keys():
            raise ValueError("SciFact 抽样包含无效查询")
        qrels = {query_id: labels for query_id, labels in qrels.items() if query_id in query_ids}

    begun = perf_counter()
    with TemporaryDirectory(prefix="context-lab-scifact-") as folder:
        rag = RAGTool(Path(folder))
        for doc_id, (title, body) in corpus.items():
            if not doc_id.isascii() or not doc_id.isdecimal():
                raise ValueError("SciFact 文档 ID 必须为 ASCII 数字")
            relative = Path(f"{doc_id}.md")
            (Path(folder) / relative).write_text(f"# {title}\n\n{body}\n", encoding="utf-8")
            rag.add_document(relative)
    chunks = rag.chunks
    chunking_ms = (perf_counter() - begun) * 1000
    if progress:
        progress(f"SciFact · {len(corpus)} 篇文档、{len(chunks)} 个文档块、{len(qrels)} 道测试题")
    texts = {chunk.id: index_text(chunk) for chunk in chunks}
    split = lru_cache(maxsize=len(texts) + len(qrels))(tokenize)
    begun = perf_counter()
    matrix = vectors.prepare(chunks) if vectors is not None else None
    vector_index_ms = (perf_counter() - begun) * 1000 if vectors is not None else None
    if vectors is not None and progress:
        progress("SciFact · 文档向量已准备完成")

    def evaluate_query(item):
        query_id, labels = item
        query = queries[query_id]
        begun = perf_counter()
        lexical = [SearchHit(chunk, score) for chunk, score in bm25_rank(
            query, chunks, candidates, key=lambda chunk: chunk.id, text=lambda chunk: texts[chunk.id], split=split)]
        lexical_ms = (perf_counter() - begun) * 1000
        rankings = {"bm25": (lexical, lexical_ms)}
        rerank_pool, rerank_ms = lexical, lexical_ms
        if vectors is not None:
            begun = perf_counter()
            dense = vectors.search(query, chunks, matrix, candidates)
            dense_ms = (perf_counter() - begun) * 1000
            rankings["vector"] = (dense, dense_ms)
            begun = perf_counter()
            hybrid = fuse_benchmark_hits(lexical, dense, candidates)
            rankings["hybrid"] = (hybrid, lexical_ms + dense_ms + (perf_counter() - begun) * 1000)
            rerank_pool, rerank_ms = dense, dense_ms
        if reranker is not None:
            pool = rerank_pool[:rerank_candidates]
            begun = perf_counter()
            reranked = rerank_benchmark_hits(reranker, query, pool, len(pool))
            rankings["vector_rerank" if vectors is not None else "bm25_rerank"] = (
                reranked, rerank_ms + (perf_counter() - begun) * 1000)
        if comparisons:
            compare_pool, compare_ms = rankings[comparison_source]
            rankings.update(compare_rerankers(query, compare_pool, compare_ms, comparisons, source=comparison_source))
        return {name: {"query_id": query_id, "relevant_ids": labels, "latency_ms": ms,
                       **_score_documents(hits, labels, k)}
                for name, (hits, ms) in rankings.items()}

    rows: dict[str, list[dict]] = defaultdict(list)
    with (ThreadPoolExecutor(rerank_workers) if rerank_workers > 1 else nullcontext()) as pool:
        results = pool.map(evaluate_query, qrels.items()) if pool is not None else map(evaluate_query, qrels.items())
        for number, result in enumerate(results, 1):
            for name, row in result.items():
                rows[name].append(row)
            if progress and number % 25 == 0:
                progress(f"SciFact · 已评测 {number}/{len(qrels)} 题")

    runs = {}
    for name, details in rows.items():
        count = len(details)
        runs[name] = {f"{metric}_at_k": sum(row[metric] for row in details) / count
                      for metric in ("hit", "recall", "ndcg")}
        runs[name].update(mrr_at_k=sum(row["rr"] for row in details) / count,
                          p95_ms=sorted(row["latency_ms"] for row in details)[ceil(0.95 * count) - 1],
                          details=details)
    return {"dataset": "BEIR SciFact", "sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
            "split": "test", "cases": len(qrels), "documents": len(corpus), "chunks": len(chunks),
            "chunk_size": rag.chunk_size, "chunk_overlap": rag.chunk_overlap, "chunking_ms": chunking_ms,
            "k": k, "candidate_limit": candidates, "vector_index_ms": vector_index_ms,
            "embedding_model": vectors.client.config.model if vectors is not None else None,
            "rerank_model": reranker.config.model if reranker is not None else None,
            "rerank_candidates": rerank_candidates if reranker is not None else None, "runs": runs}
