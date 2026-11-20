"""为固定评测语料缓存已有 HTTP 编码器的归一化向量。"""

import hashlib
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from time import sleep

import numpy as np

from ..embeddings.errors import EmbeddingRequestError
from ..types import DocumentChunk, SearchHit
from .recall import index_text


class CachedBenchmarkVectors:
    MAX_CHARS = 8192

    def __init__(self, client, cache_path: Path, *, workers: int = 1):
        self.client = client
        self.cache_path = cache_path
        self.workers = workers
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.profile = f"normalized-f32-v1-{self.MAX_CHARS}:{client.config.profile_id}"

    def prepare(self, chunks: list[DocumentChunk]):
        """按模型和实际发送的文本命中缓存，每成功一批即写入 SQLite。"""
        texts = [index_text(chunk)[:self.MAX_CHARS] for chunk in chunks]
        keys = [hashlib.sha256(text.encode()).hexdigest() for text in texts]
        missing = dict(zip(keys, texts))
        keys_unique = list(missing)
        vectors = {}
        with closing(sqlite3.connect(self.cache_path)) as cache:
            cache.execute("""CREATE TABLE IF NOT EXISTS vectors (
                profile TEXT, digest TEXT, embedding BLOB NOT NULL, PRIMARY KEY(profile, digest))""")
            for start in range(0, len(keys_unique), 500):
                batch_keys = keys_unique[start:start + 500]
                placeholders = ",".join("?" for _ in batch_keys)
                rows = cache.execute(f"SELECT digest, embedding FROM vectors WHERE profile=? AND digest IN ({placeholders})",
                                     (self.profile, *batch_keys))
                for key, data in rows:
                    vectors[key] = np.frombuffer(data, dtype="<f4")
                    missing.pop(key)
            items = list(missing.items())
            size = self.client.config.batch_size
            batches = [items[start:start + size] for start in range(0, len(items), size)]
            if batches:
                with ThreadPoolExecutor(max_workers=self.workers) as pool:
                    # 只提交一小组并行任务，接口持续失败时不继续请求整个仓库。
                    for start in range(0, len(batches), self.workers):
                        group = batches[start:start + self.workers]
                        for batch, batch_vectors in zip(group, pool.map(self._embed_batch, group)):
                            rows = []
                            for (key, _), vector in zip(batch, batch_vectors):
                                vectors[key] = vector
                                rows.append((self.profile, key, vector.tobytes()))
                            with cache:
                                cache.executemany("INSERT OR REPLACE INTO vectors VALUES (?, ?, ?)", rows)
        matrix = np.stack([vectors[key] for key in keys])
        if self.client.dimensions not in (None, matrix.shape[1]):
            raise ValueError("缓存向量维度与当前模型不一致")
        return matrix

    def _embed_batch(self, batch: list[tuple[str, str]]):
        texts = [text for _, text in batch]
        for attempt in range(3):
            try:
                return self._normalized(self.client.embed(texts))
            except EmbeddingRequestError as exc:
                if exc.error_code == "http_400" and len(batch) > 1:
                    middle = len(batch) // 2
                    return np.concatenate((self._embed_batch(batch[:middle]), self._embed_batch(batch[middle:])))
                if not exc.retryable or attempt == 2:
                    raise
                sleep(2 ** attempt)

    @staticmethod
    def _normalized(vectors):
        matrix = np.asarray(vectors, dtype=np.float64)
        return (matrix / np.linalg.norm(matrix, axis=1, keepdims=True)).astype("<f4")

    def search(self, query: str, chunks: list[DocumentChunk], matrix, limit: int) -> list[SearchHit]:
        return self.rank(chunks, self.score_queries([query], matrix)[0], limit)

    def score_queries(self, queries: list[str], matrix):
        """批量生成候选的原始余弦分数，用于各查询变体的融合。"""
        query_vectors = self._embed_batch([("", query[:self.MAX_CHARS]) for query in queries])
        if matrix.shape[1] != query_vectors.shape[1]:
            raise ValueError("查询向量与缓存的向量维度不一致")
        return [matrix @ vector for vector in query_vectors]

    @staticmethod
    def rank(chunks: list[DocumentChunk], scores, limit: int) -> list[SearchHit]:
        indices = sorted(range(len(chunks)), key=lambda index: (-scores[index], chunks[index].id))[:limit]
        return [SearchHit(chunks[index], float(scores[index]), vector_score=float(scores[index])) for index in indices]
