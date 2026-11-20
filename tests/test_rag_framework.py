"""已实现的 RAG 接线验收，分块和高级策略有各自的练习测试。"""

import hashlib
import json
from unittest.mock import Mock, patch

import httpx

from context_lab.embeddings import EmbeddingHit, IndexedChunk, IndexStatus
from context_lab.rag import RAGSearchOptions
from context_lab.rag.embedding_bridge import RAGEmbeddingBridge, RAGIndexMismatch
from context_lab.rag.recall import index_text
from context_lab.rag.rerank import ECNUReranker, RerankConfig, RerankRequestError
from context_lab.tools.rag import RAGTool
from context_lab.types import DocumentChunk, SearchHit
from tests.support import CharacterCounter, SandboxTestCase


def indexed(chunk, namespace):
    text = index_text(chunk)
    return IndexedChunk(
        "profile", namespace, chunk.source, chunk.id, text, hashlib.sha256(text.encode()).hexdigest(),
        {**chunk.metadata, "heading_path": chunk.heading_path, "start": chunk.start, "end": chunk.end}, "ready",
    )


class RAGFrameworkTests(SandboxTestCase):
    def test_default_chunks_leave_room_for_labels_in_text_model_requests(self):
        tool = RAGTool(self.tmp)
        path = self.tmp / "long.md"
        path.write_text("# " + "heading" * 500 + "\n\n" + "x" * 14_000, encoding="utf-8")
        chunks = tool.add_document(path)

        self.assertGreater(len(chunks), 2)
        self.assertTrue(all(len(index_text(chunk)) <= 8192 for chunk in chunks))
        self.assertTrue(all(left.end > right.start for left, right in zip(chunks, chunks[1:])))

    def test_scan_keeps_preamble_and_same_named_sections_separate(self):
        tool = RAGTool(self.tmp)
        text = "前言\r\n\r\n## 配置\r\n甲\r\n## 配置\r\n乙"

        sections = tool._split_paragraphs_with_headings(text)

        self.assertEqual(sections, [
            {"heading_path": None, "start": 0, "end": 6, "paragraphs": [("前言", 0, 2)]},
            {"heading_path": "配置", "start": 6, "end": 16, "paragraphs": [("甲", 13, 14)]},
            {"heading_path": "配置", "start": 16, "end": 24, "paragraphs": [("乙", 23, 24)]},
        ])

    def test_ingestion_keeps_raw_offsets_and_stable_identity_after_move(self):
        path = self.tmp / "manual.md"
        tool = RAGTool(self.tmp, chunk_size=60, chunk_overlap=0, token_counter=CharacterCounter())
        text = "## 缓存\r\n\r\n缓存处理"
        path.write_bytes(text.encode())
        start = text.index("缓存处理")
        with patch.object(
            tool, "_chunk_paragraphs",
            return_value=[DocumentChunk("", "", text[start:], "缓存", start, len(text))],
        ):
            first, = tool.add_document(path)
        self.assertEqual(first.metadata["start_line"], 3)
        self.assertEqual(first.metadata["parent_start"], 0)
        self.assertEqual(first.metadata["parent_end"], len(text))
        self.assertEqual(tool.sources[first.source][first.start:first.end], first.content)

        moved = "\r\n" + text
        path.write_bytes(moved.encode())
        with patch.object(
            tool, "_chunk_paragraphs",
            return_value=[DocumentChunk("", "", first.content, "缓存", first.start + 2, len(moved))],
        ):
            second, = tool.add_document(path)
        self.assertEqual(first.id, second.id)
        self.assertEqual(second.metadata["start_line"], 4)
        self.assertEqual(second.metadata["parent_start"], 2)
        self.assertEqual(second.metadata["parent_end"], len(moved))
        self.assertNotEqual(first.metadata["source_hash"], second.metadata["source_hash"])
        self.assertEqual(tool.chunks, [second])

    def test_scope_and_hybrid_search_keep_original_blocks_and_raw_scores(self):
        index = Mock()
        tool = RAGTool(self.tmp, repo_id="repo", revision="rev1", embedding_index=index)
        chunk = DocumentChunk("a", "src/a.py", "def get_cached(): pass", metadata={"language": "python"})
        wrong_prefix = DocumentChunk("b", "src_old/a.py", chunk.content, metadata={"language": "python"})
        wrong_language = DocumentChunk("c", "src/b.md", chunk.content, metadata={"language": "markdown"})
        tool.chunks = [chunk, wrong_prefix, wrong_language]
        with patch.object(tool._vector, "search", return_value=[SearchHit(chunk, 0.7)]) as search:
            hit, = tool.search("get_cached", 1, options=RAGSearchOptions(
                mode="hybrid", path_prefix="src", language="python",
            ))
        search.assert_called_once()
        self.assertEqual(search.call_args.args[1], [chunk])
        self.assertIs(hit.chunk, chunk)
        self.assertGreater(hit.lexical_score, 0)
        self.assertEqual(hit.vector_score, 0.7)
        self.assertEqual(hit.ranks, {"lexical": 1, "vector": 1})
        self.assertEqual(hit.fusion_score, hit.score)

    def test_vector_bridge_restores_and_expands_a_filtered_window(self):
        index = Mock()
        index.profile_id = "profile"
        index.status.return_value = IndexStatus(ready=5)
        index.client.embed.return_value = [[1.0, 2.0]]
        bridge = RAGEmbeddingBridge(index, "repo", "rev")
        text = "cache"
        source_hash = hashlib.sha256(text.encode()).hexdigest()
        a = DocumentChunk("a", "src/a.py", text, start=0, end=len(text), metadata={"source_hash": source_hash})
        b = DocumentChunk("b", "src/b.py", text, start=0, end=len(text), metadata={"source_hash": source_hash})
        # Limit 1 初次取 4 条，真正允许的块在第 5 条。
        others = [DocumentChunk(str(i), "outside.py", text) for i in range(4)]
        stored_hits = [EmbeddingHit(indexed(chunk, bridge.namespace), 0.9) for chunk in [*others, b]]
        index.store.search_ready.side_effect = lambda *args, limit, source_id: stored_hits[:limit]
        hit, = bridge.search("cache", [a, b], 1)
        self.assertIs(hit.chunk, b)
        index.client.embed.assert_called_once_with(["cache"])
        self.assertEqual([call.kwargs["limit"] for call in index.store.search_ready.call_args_list], [4, 5])

        bridge.sync_source(a.source, [a])
        namespace, source, blocks = index.sync_source.call_args.args
        self.assertEqual((namespace, source, blocks[0].id), ("rag:repo:rev", a.source, a.id))
        self.assertEqual(blocks[0].text, index_text(a))
        index.store.list_source.return_value = [indexed(a, bridge.namespace)]
        self.assertEqual(bridge.restore_source(a.source, text), [a])
        with self.assertRaises(RAGIndexMismatch):
            bridge.restore_source(a.source, "changed")

    def test_ecnu_rerank_request_and_index_mapping(self):
        left = SearchHit(DocumentChunk("left", "a.py", "x" * 9000), 0.2, lexical_score=2.0)
        right = SearchHit(DocumentChunk("right", "b.py", "cache"), 0.1, vector_score=0.5, fusion_score=0.1)

        def handle(request):
            self.assertEqual(str(request.url), "https://chat.ecnu.edu.cn/open/api/v1/rerank")
            self.assertEqual(request.headers["Authorization"], "Bearer offline-test")
            payload = json.loads(request.content)
            self.assertEqual((payload["model"], payload["top_n"], payload["return_documents"]),
                             ("ecnu-rerank", 2, False))
            self.assertEqual(payload["query"], "cache")
            self.assertEqual(len(payload["documents"][0]), 8192)
            return httpx.Response(200, json={"results": [
                {"index": 0, "relevance_score": 0.2}, {"index": 1, "relevance_score": 0.9},
            ]})

        with httpx.Client(transport=httpx.MockTransport(handle)) as client:
            ranked = ECNUReranker(RerankConfig(api_key="offline-test"), client).rerank("cache", [left, right], 2)
        self.assertEqual([hit.chunk.id for hit in ranked], ["right", "left"])
        self.assertIs(ranked[1].chunk, left.chunk)
        self.assertEqual((ranked[0].rerank_score, ranked[0].vector_score, ranked[1].lexical_score), (0.9, 0.5, 2.0))
        self.assertEqual(ranked[0].fusion_score, 0.1)
        self.assertEqual(len(ranked[1].chunk.content), 9000)

    def test_rerank_rejects_invalid_indexes(self):
        for indexes in ([0, 0], [0, 2], [0, True]):
            with self.subTest(indexes=indexes), self.assertRaises(RerankRequestError):
                ECNUReranker._parse_response(
                    {"results": [{"index": index, "relevance_score": 0.1} for index in indexes]}, 2, 2,
                )

    def test_configured_rerank_runs_on_recall_pool_through_normal_search(self):
        reranker = Mock()
        tool = RAGTool(self.tmp, reranker=reranker,
                       search_options=RAGSearchOptions(candidate_limit=2, rerank=True))
        tool.chunks = [
            DocumentChunk("a", "a.py", "cache"), DocumentChunk("b", "b.py", "cache lookup"),
            DocumentChunk("c", "c.py", "unrelated"),
        ]
        reranker.rerank.side_effect = lambda query, hits, limit: [hits[-1]]
        result = tool.search("cache", limit=1)
        query, pool, limit = reranker.rerank.call_args.args
        self.assertEqual((query, limit), ("cache", 1))
        self.assertEqual({hit.chunk.id for hit in pool}, {"a", "b"})
        self.assertEqual(result, [pool[-1]])
