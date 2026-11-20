"""E04：只验本阶段承诺的文本分块与词法检索。"""

from context_lab.tools.rag import RAGTool
from context_lab.types import DocumentChunk, SearchHit
from tests.support import CharacterCounter, SandboxTestCase


class RetrievalTests(SandboxTestCase):
    def make_tool(self, **kwargs):
        kb = self.tmp / "knowledge"
        kb.mkdir(exist_ok=True)
        return RAGTool(kb, token_counter=CharacterCounter(), **kwargs)

    def test_chunks_preserve_original_character_offsets_and_heading(self):
        tool = self.make_tool(chunk_size=100, chunk_overlap=20)
        source = "# 运维手册\n\n## 缓存\n\n中文 redis timeout。\n\n检查连接池。\n\n## 磁盘\n\n检查 disk 使用量。\n"
        path = self.tmp / "runbook.md"
        path.write_text(source, encoding="utf-8")
        chunks = tool.add_document(path)
        self.assertTrue(chunks)
        self.assertEqual(len({c.id for c in chunks}), len(chunks))
        for chunk in chunks:
            self.assertIsInstance(chunk, DocumentChunk)
            self.assertGreaterEqual(chunk.start, 0)
            self.assertGreater(chunk.end, chunk.start)
            self.assertEqual(source[chunk.start:chunk.end], chunk.content)
            self.assertLessEqual(len(chunk.content), tool.chunk_size)
            self.assertTrue(chunk.source)
        cache_chunks = [c for c in chunks if "redis" in c.content]
        self.assertTrue(any("缓存" in (c.heading_path or "") for c in cache_chunks))

    def test_overlap_reuses_a_complete_paragraph(self):
        tool = self.make_tool(chunk_size=75, chunk_overlap=35)
        paragraphs = [f"段落{i} " + chr(65 + i) * 24 + "。" for i in range(5)]
        source = "\n\n".join(paragraphs)
        path = self.tmp / "paragraphs.txt"
        path.write_text(source, encoding="utf-8")
        chunks = tool.add_document(path)
        self.assertGreater(len(chunks), 1)
        shared_paragraphs = [
            p for left, right in zip(chunks, chunks[1:]) for p in paragraphs
            if p in left.content and p in right.content
        ]
        self.assertTrue(shared_paragraphs, "重叠应保留完整尾部段落，而不是任意字符后缀")

    def test_long_paragraph_overlaps_on_complete_sentences(self):
        tool = self.make_tool(chunk_size=8, chunk_overlap=4)
        path = self.tmp / "sentences.txt"
        path.write_text("甲甲。乙乙乙。丙丙。", encoding="utf-8")

        chunks = tool.add_document(path)

        self.assertEqual(
            [(chunk.content, chunk.start, chunk.end) for chunk in chunks],
            [("甲甲。乙乙乙。", 0, 7), ("乙乙乙。丙丙。", 3, 10)],
        )

    def test_sentence_over_budget_uses_token_windows(self):
        tool = self.make_tool(chunk_size=4, chunk_overlap=1)
        path = self.tmp / "long_sentence.txt"
        path.write_text("开始。甲乙丙丁戊己庚。结束。", encoding="utf-8")

        chunks = tool.add_document(path)

        self.assertEqual(
            [(chunk.content, chunk.start, chunk.end) for chunk in chunks],
            [
                ("开始。", 0, 3),
                ("甲乙丙丁", 3, 7),
                ("丁戊己庚", 6, 10),
                ("庚。", 9, 11),
                ("结束。", 11, 14),
            ],
        )

    def test_plain_text_keeps_repeated_paragraph_offsets_without_headings(self):
        tool = self.make_tool(chunk_size=12, chunk_overlap=0)
        source = "相同段落。\r\n\r\n相同段落。\r\n\r\n最后一段。"
        path = self.tmp / "plain.txt"
        path.write_bytes(source.encode("utf-8"))
        chunks = tool.add_document(path)
        repeated = [chunk for chunk in chunks if chunk.content == "相同段落。"]
        self.assertEqual([chunk.start for chunk in repeated], [0, source.index("相同段落。", 1)])
        for chunk in chunks:
            self.assertIsNone(chunk.heading_path)
            self.assertEqual(source[chunk.start:chunk.end], chunk.content)

    def test_long_paragraph_uses_injected_token_budget_without_losing_text(self):
        class DoubleCharacterCounter(CharacterCounter):
            def count(self, text):
                self.calls.append(text)
                return 2 * len(text)

            def truncate(self, text, max_tokens):
                if max_tokens < 0:
                    raise ValueError("预算不能为负")
                return text[:max_tokens // 2]

        source = "甲乙丙丁" * 23
        path = self.tmp / "long.txt"
        path.write_text(source, encoding="utf-8")

        for overlap, expected_overlap in ((0, 0), (3, 2), (19, 18)):
            with self.subTest(overlap=overlap):
                counter = DoubleCharacterCounter()
                tool = RAGTool(self.tmp, chunk_size=20, chunk_overlap=overlap, token_counter=counter)
                chunks = tool.add_document(path)

                covered = set()
                self.assertGreater(len(chunks), 1)
                for chunk in chunks:
                    self.assertEqual(source[chunk.start:chunk.end], chunk.content)
                    self.assertTrue(chunk.content)
                    self.assertLessEqual(counter.count(chunk.content), 20)
                    covered.update(range(chunk.start, chunk.end))
                self.assertEqual(covered, set(range(len(source))), "单段超预算仍须保留完整原文覆盖")
                for previous, current in zip(chunks, chunks[1:]):
                    self.assertGreater(current.end, previous.end)
                    self.assertEqual(counter.count(source[current.start:previous.end]), expected_overlap)

    def test_search_returns_relevant_typed_hits_with_limit(self):
        tool = self.make_tool()
        for name, text in [("cache.md", "# 缓存\n\nredis timeout connection"), ("other.txt", "meeting agenda lunch")]:
            path = self.tmp / name
            path.write_text(text, encoding="utf-8")
            tool.add_document(path)
        hits = tool.search("redis timeout", limit=1)
        self.assertEqual(len(hits), 1)
        self.assertIsInstance(hits[0], SearchHit)
        self.assertIn("redis", hits[0].chunk.content)
        self.assertGreater(hits[0].score, 0)

    def test_pdf_is_rejected_instead_of_pretending_to_parse_it(self):
        tool = self.make_tool()
        path = self.tmp / "unsupported.pdf"
        path.write_bytes(b"%PDF-1.4\nnot a text document")
        with self.assertRaises(ValueError):
            tool.add_document(path)
