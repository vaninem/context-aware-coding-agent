"""RAG 高级练习验收，尚未实现的核心函数会抛出 ExercisePending。"""

from unittest.mock import patch

from context_lab.context import ContextBuilder
from context_lab.models import ScriptedChatModel
from context_lab.rag.chunking import split_python
from context_lab.rag.evaluation import compare_vector_indexes, evaluate_retrieval
from context_lab.rag.expansion import build_symbol_index, expand_dependencies, expand_parents, retrieve_iteratively
from context_lab.rag.query import plan_query
from context_lab.rag.scifact import _score_documents
from context_lab.rag.source_coverage import covered_lines, target_lines
from context_lab.rag.types import RAGSearchOptions, RetrievalCase
from context_lab.tools.rag import RAGTool
from context_lab.types import DocumentChunk, SearchHit
from tests.support import CharacterCounter, SandboxTestCase


class RAGExerciseTests(SandboxTestCase):
    def test_source_coverage_reassembles_split_lines_without_charging_for_duplicate_source(self):
        class CharacterEncoding:
            encode_ordinary = staticmethod(list)

            @staticmethod
            def decode_bytes(characters):
                return "".join(characters).encode()

        text = "header\ndef f():\n    return 1\nfooter"
        source, noise = "target.py", "noise.py"
        start = text.index("def")
        split = text.index("return") + 3
        end = text.index("\nfooter")
        first = SearchHit(DocumentChunk("first", source, text[start:split], start=start, end=split), 1)
        second = SearchHit(DocumentChunk("second", source, text[split - 2:end], start=split - 2, end=end), 1)
        wrong = SearchHit(DocumentChunk("wrong", noise, "!", start=0, end=1), 1)
        budget = end - start
        lines = target_lines(text, start, end)
        sources = {source: text, noise: "!"}

        self.assertEqual(len(lines), 2)
        self.assertEqual(covered_lines([first, first, second], sources, source, lines, budget, CharacterEncoding()), 2)
        self.assertEqual(covered_lines([wrong, first, second], sources, source, lines, budget, CharacterEncoding()), 1)

    def test_document_benchmark_deduplicates_chunks_before_scoring_top_k(self):
        hits = [SearchHit(DocumentChunk(str(i), f"{source}.md", "text"), 1)
                for i, source in enumerate(("1", "1", "2"))]
        score = _score_documents(hits, {"2": 1, "3": 1}, 2)
        self.assertEqual(score["top_document_ids"], ["1", "2"])
        self.assertEqual(score["first_relevant_rank"], 2)
        self.assertEqual(score["recall"], 0.5)
        self.assertEqual(score["rr"], 0.5)

    def test_python_chunks_keep_raw_offsets_symbol_and_parent(self):
        text = 'class 缓存:\n    def get_cached(self):\n        return fetch()\n'
        chunks = split_python("src/cache.py", text, 80, CharacterCounter())
        method = next(chunk for chunk in chunks if "get_cached" in chunk.metadata.get("symbol", ""))
        self.assertIn("fetch", method.metadata["references"])
        for chunk in chunks:
            self.assertEqual(chunk.content, text[chunk.start:chunk.end])
            self.assertLessEqual(len(chunk.content), 80)
            self.assertLessEqual(chunk.metadata["parent_start"], chunk.start)
            self.assertGreaterEqual(chunk.metadata["parent_end"], chunk.end)

    def test_python_ingestion_tracks_global_writes_and_class_method_ownership(self):
        text = (
            "import os\r\n"
            "from storage import load_cache as load\r\n"
            "count = 10\r\n"
            "count += 1\r\n"
            "import json\r\n"
            "class 缓存:\r\n"
            "    @staticmethod\r\n"
            "    def read():\r\n"
            '        return load(count, "默认")\r\n'
            "    def bump(self):\r\n"
            "        global count\r\n"
            "        count += 1\r\n"
            "    def local(self, count):\r\n"
            "        count += 1\r\n"
            "        return count\r\n"
        )
        path = self.tmp / "cache.py"
        path.write_bytes(text.encode("utf-8"))
        tool = RAGTool(
            self.tmp, chunk_size=100, chunk_overlap=0, token_counter=CharacterCounter(), parent_max_tokens=400,
        )

        chunks = tool.add_document(path)

        writes = [write for chunk in chunks for write in chunk.metadata.get("global_writes", ())]
        self.assertEqual(
            [(write["name"], write["operation"], text[write["start"]:write["end"]]) for write in writes],
            [("count", "assign", "count = 10"), ("count", "augment", "count += 1"),
             ("count", "augment", "count += 1")],
        )
        self.assertEqual(chunks[0].content, "import os\r\nfrom storage import load_cache as load")
        self.assertIn("import json", [chunk.content for chunk in chunks])
        methods = {chunk.metadata["symbol"]: chunk for chunk in chunks if "class_name" in chunk.metadata}
        read = methods["read"]
        self.assertEqual(read.metadata["qualified_symbol"], "cache.缓存.read")
        self.assertEqual(read.metadata["references"], ["count", "storage.load_cache"])
        self.assertTrue(read.content.lstrip().startswith("@staticmethod"))
        for method in methods.values():
            self.assertEqual(method.metadata["class_name"], "缓存")
            self.assertEqual(method.metadata["parent_start"], text.index("class 缓存"))
            self.assertEqual(method.metadata["parent_end"], len(text.rstrip()))
        self.assertNotIn("count", methods["local"].metadata["references"])
        self.assertNotIn("global_writes", methods["local"].metadata)
        retrieved = {hit.chunk.content for hit in tool.search("count", limit=len(chunks))}
        self.assertTrue({"count = 10", "count += 1"} <= retrieved)

    def test_python_long_function_uses_statement_boundaries_and_keeps_raw_coverage(self):
        text = 'def work():\n    first = 1\n    message = "缓存缓存缓存缓存缓存"\n    return first'
        chunks = split_python("src/tasks.py", text, 28, CharacterCounter())

        self.assertEqual(chunks[0].end, text.index("1") + 1)
        self.assertEqual("".join(chunk.content for chunk in chunks), text)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.content), 28)
            self.assertEqual(text[chunk.start:chunk.end], chunk.content)
            self.assertEqual(chunk.metadata["qualified_symbol"], "src.tasks.work")
            self.assertEqual(chunk.metadata["parent_end"], len(text))

    def test_python_ingestion_limits_large_class_parents_at_method_boundaries(self):
        text = (
            "class Client:\n"
            "    def first(self):\n        a = 1\n        return a\n"
            "    def second(self):\n        b = 2\n        return b\n"
        )
        path = self.tmp / "client.py"
        path.write_text(text, encoding="utf-8")
        tool = RAGTool(self.tmp, chunk_size=32, chunk_overlap=0, token_counter=CharacterCounter(), parent_max_tokens=64)

        chunks = tool.add_document(path)

        for method in ("first", "second"):
            children = [chunk for chunk in chunks if chunk.metadata.get("symbol") == method]
            self.assertGreater(len(children), 1)
            bounds = {(child.metadata["parent_start"], child.metadata["parent_end"]) for child in children}
            self.assertEqual(len(bounds), 1)
            (start, end), = bounds
            self.assertLessEqual(end - start, 64)
            self.assertIn(f"def {method}(self):", text[start:end])
            for child in children:
                self.assertTrue(start <= child.start < child.end <= end)

    def test_query_plan_keeps_exact_original_and_hyde_on_vector_route(self):
        model = ScriptedChatModel([
            '{"queries":["get_cached HTTP_429","get_cached HTTP_429"],"hypothetical_document":"调用限流后的缓存回退"}',
        ])
        query = "为什么 HTTP_429 会触发重试"
        plan = plan_query(query, model, rewrite=True, hyde=True)
        self.assertEqual(plan.lexical_queries[0], query)
        self.assertEqual(plan.vector_queries[0], query)
        self.assertEqual(plan.lexical_queries.count("get_cached HTTP_429"), 1)
        self.assertIn("调用限流后的缓存回退", plan.vector_queries)
        self.assertNotIn("调用限流后的缓存回退", plan.lexical_queries)

    def test_parent_expansion_keeps_best_child_as_citation(self):
        lines = ("load()", "check()", "store()", "return 1")
        text = "def get_cached():\n" + "".join(f"    {line}\n" for line in lines)
        bounds = {"parent_start": 0, "parent_end": len(text)}
        children = []
        for i, line in enumerate(lines):
            start = text.index(line)
            children.append(DocumentChunk(str(i), "cache.py", line, start=start, end=start + len(line), metadata=bounds))
        hits = [SearchHit(child, 4 - i) for i, child in enumerate(children)]
        sources = {"cache.py": text}
        counter = CharacterCounter()

        merged = expand_parents(hits[:3], sources, len(text), counter, candidates=children)
        self.assertEqual(len(merged), 1)
        self.assertIs(merged[0].chunk, children[0])
        self.assertEqual(merged[0].context_chunk.content, text)
        self.assertEqual(merged[0].score, hits[0].score)

        for visible_hits, budget in ((hits[:2], len(text)), (hits[:3], len(lines[0]) + len(lines[1]))):
            with self.subTest(visible_hits=len(visible_hits), budget=budget):
                expanded = expand_parents(visible_hits, sources, budget, counter, candidates=children)
                self.assertEqual(expanded, hits[:2])

    def test_mixed_search_expands_code_and_returns_original_document_chunks(self):
        tool = RAGTool(
            self.tmp, chunk_size=56, chunk_overlap=0, parent_max_tokens=200,
            token_counter=CharacterCounter(), search_options=RAGSearchOptions(expand_parents=True),
        )
        code = "class Client:\n" + "".join(
            f'    def {name}(self):\n        return "needle"\n' for name in ("first", "second", "third")
        )
        (self.tmp / "client.py").write_text(code, encoding="utf-8")
        (self.tmp / "guide.md").write_text(
            "## Policy\n\nneedle is required for a retried request.\n\n"
            "needle is checked before opening a connection.\n", encoding="utf-8",
        )
        tool.add_document(self.tmp / "client.py")
        document_chunks = tool.add_document(self.tmp / "guide.md")

        hits = tool.search("needle")
        code_hit, = [hit for hit in hits if hit.chunk.source == "client.py"]
        self.assertEqual(code_hit.context_chunk.content, code.rstrip())
        self.assertIn('return "needle"', code_hit.chunk.content)
        document_hits = [hit for hit in hits if hit.chunk.source == "guide.md"]
        self.assertEqual({hit.chunk.id for hit in document_hits}, {chunk.id for chunk in document_chunks})
        self.assertTrue(all(hit.context_chunk is None for hit in document_hits))

        packets = ContextBuilder(rag_tool=tool).collect_sources("needle")
        self.assertEqual(
            {packet.content for packet in packets if packet.metadata["source"] == "guide.md"},
            {chunk.content for chunk in document_chunks},
        )

    def test_symbol_index_keeps_homonyms_and_qualified_names(self):
        one = DocumentChunk("a", "one.py", "def read(): pass",
                            metadata={"symbol": "read", "qualified_symbol": "one.read"})
        two = DocumentChunk("b", "two.py", "def read(): pass",
                            metadata={"symbol": "read", "qualified_symbol": "two.read"})
        index = build_symbol_index([one, two, one])
        self.assertEqual(len(index["read"]), 2)
        self.assertEqual(index["one.read"], [one])

    def test_dependency_expansion_adds_linked_definition_within_limit(self):
        caller = DocumentChunk("a", "app.py", "storage.load_cache()", metadata={
            "references": ["storage.load_cache", "load_cache", "storage.clear_cache"],
        })
        target = DocumentChunk("b", "storage.py", "def load_cache(): pass")
        other = DocumentChunk("c", "storage.py", "def clear_cache(): pass")
        symbols = {"storage.load_cache": [target], "load_cache": [target], "storage.clear_cache": [other]}
        original = SearchHit(caller, 1)

        hits = expand_dependencies([original], symbols, limit=2)
        self.assertEqual([hit.chunk.id for hit in hits], ["a", "b"])
        self.assertIs(hits[0], original)
        self.assertEqual(hits[1].chunk.metadata["dependency_from_chunk_id"], "a")
        self.assertEqual(hits[1].chunk.metadata["dependency_reference"], "storage.load_cache")
        self.assertEqual(target.metadata, {})
        hits = expand_dependencies([original], symbols, limit=3)
        self.assertEqual([hit.chunk.id for hit in hits], ["a", "b", "c"])

    def test_dependency_search_reuses_original_hit_and_score_for_a_recalled_definition(self):
        caller = DocumentChunk("a", "app.py", "storage.load_cache()", metadata={"references": ["storage.load_cache"]})
        target = DocumentChunk("b", "storage.py", "def load_cache(): pass", metadata={
            "symbol": "load_cache", "qualified_symbol": "storage.load_cache",
        })
        original = [SearchHit(caller, 0.9), SearchHit(target, 0.7, lexical_score=0.7)]
        tool = RAGTool(self.tmp)
        tool.chunks = [caller, target]

        with patch.object(tool, "_recall", return_value=original):
            hits = tool.search("load_cache", limit=2, options=RAGSearchOptions(follow_dependencies=True))

        self.assertEqual(len(hits), 2)
        self.assertIs(hits[0], original[0])
        self.assertIs(hits[1], original[1])

    def test_iterative_search_uses_new_symbol_and_preserves_initial_hit(self):
        caller = SearchHit(DocumentChunk("a", "app.py", "load_cache()", metadata={"references": ["load_cache"]}), 1)
        target = SearchHit(DocumentChunk("b", "storage.py", "def load_cache(): pass"), 0.8)
        queries = []

        def lookup(query):
            queries.append(query)
            return [caller, target]

        hits = retrieve_iteratively("缓存加载失败", [caller], lookup, max_rounds=2, limit=2)
        self.assertEqual(len(queries), 1)
        self.assertIn("load_cache", queries[0])
        self.assertEqual({hit.chunk.id for hit in hits}, {"a", "b"})
        self.assertEqual(len(hits), 2)

    def test_iterative_search_uses_only_new_evidence_and_stops_on_repeated_hits(self):
        caller = SearchHit(DocumentChunk("a", "app.py", "load_cache()", metadata={
            "references": ["load_cache", "old_hint"],
        }), 1.0, lexical_score=1.0)
        other = SearchHit(DocumentChunk("b", "app.py", "read()", metadata={"references": ["unused"]}), 0.8)
        target = SearchHit(DocumentChunk("c", "storage.py", "on_error()", metadata={
            "references": ["load_cache", "on_error"],
        }), 0.7)
        queries = []

        def lookup(query):
            queries.append(query)
            return [caller, target, target] if query.endswith("load_cache") else [target, caller]

        hits = retrieve_iteratively("缓存加载失败", [caller, caller, other], lookup, max_rounds=5, limit=3)

        self.assertEqual(queries, ["缓存加载失败 load_cache", "缓存加载失败 on_error"])
        self.assertEqual([hit.chunk.id for hit in hits], ["a", "c", "b"])
        self.assertEqual(hits[0].lexical_score, 1.0)
        self.assertEqual(hits[0].score, hits[0].fusion_score)
        self.assertAlmostEqual(hits[1].score, 1 / 62 + 1 / 61)

    def test_retrieval_metrics_use_first_relevant_rank_and_total_labels(self):
        a = SearchHit(DocumentChunk("a", "a.py", "related"), 0.5)
        b = SearchHit(DocumentChunk("b", "b.py", "unrelated"), 0.8)
        metrics = evaluate_retrieval([RetrievalCase("q", frozenset({"a", "c"}))],
                                     lambda query, limit: [b, a], limit=2)
        self.assertEqual((metrics.cases, metrics.recall_at_k, metrics.mrr_at_k), (1, 0.5, 0.5))
        self.assertGreaterEqual(metrics.p95_ms, 0)

    def test_index_comparison_uses_same_queries_and_exact_reference(self):
        def hits(ids):
            return [SearchHit(DocumentChunk(i, f"{i}.py", i), 1) for i in ids]

        metrics = compare_vector_indexes(["q"], lambda query, limit: hits("ab"),
                                         lambda query, limit: hits("ac"), limit=2)
        self.assertEqual(metrics.queries, 1)
        self.assertEqual(metrics.mean_overlap_at_k, 0.5)
