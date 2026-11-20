"""E06–E08：用替身隔离检索与计数，检查上下文预算行为。"""

from datetime import timedelta
import unittest

from context_lab.config import ContextConfig
from context_lab.context import ContextBuilder
from context_lab.errors import ContextBudgetError, ExercisePending
from context_lab.memory import MemoryRecord
from context_lab.types import ContextBuildResult, ContextPacket, DocumentChunk, Message, SearchHit
from tests.support import CharacterCounter, FakeRetriever, NOW


def packet(content, kind="general", score=None, **kwargs):
    return ContextPacket(content, timestamp=kwargs.pop("timestamp", NOW), relevance_score=score,
                         metadata={"type": kind, **kwargs})


class GatherTests(unittest.TestCase):
    def test_gather_preserves_sources_and_ids(self):
        memory = FakeRetriever([MemoryRecord("memory-42", "remember retry", created_at=NOW)])
        chunk = DocumentChunk("chunk-9", "runbook.md", "timeout evidence")
        rag = FakeRetriever([SearchHit(chunk, 0.7)])
        builder = ContextBuilder(memory_tool=memory, rag_tool=rag, token_counter=CharacterCounter())
        custom = packet("my note", "note", id="note-3")
        gathered = builder._gather(
            "timeout", conversation_history=[Message("user", "earlier", NOW)],
            system_instructions="never leak secrets", custom_packets=[custom],
        )
        types = {p.metadata.get("type") for p in gathered}
        self.assertTrue({"system_instruction", "memory", "rag_result", "conversation_history", "note"} <= types)
        memory_packets = [p for p in gathered if p.metadata.get("type") == "memory"]
        rag_packets = [p for p in gathered if p.metadata.get("type") == "rag_result"]
        self.assertIn("memory-42", str([p.metadata for p in memory_packets]))
        self.assertIn("chunk-9", str([p.metadata for p in rag_packets]))
        self.assertIn(custom, gathered)
        self.assertEqual(memory.calls[0][0], "timeout")
        self.assertEqual(rag.calls[0][0], "timeout")

    def test_history_limit_zero_means_no_history(self):
        builder = ContextBuilder(ContextConfig(history_limit=0), token_counter=CharacterCounter())
        gathered = builder._gather("now", conversation_history=[Message("user", "old", NOW)])
        self.assertFalse(any(p.metadata.get("type") == "conversation_history" for p in gathered))

    def test_pending_dependency_is_not_swallowed(self):
        retriever = FakeRetriever(error=ExercisePending("M04", "not completed"))
        builder = ContextBuilder(memory_tool=retriever, token_counter=CharacterCounter())
        with self.assertRaises(ExercisePending) as caught:
            builder._gather("question")
        self.assertEqual(caught.exception.exercise_id, "M04")


class SelectionTests(unittest.TestCase):
    def builder(self, **kwargs):
        config = ContextConfig(min_relevance=0, **kwargs)
        return ContextBuilder(config, token_counter=CharacterCounter())

    def test_relevance_and_recency_have_meaningful_order(self):
        builder = self.builder()
        relevant = builder._calculate_relevance("redis timeout cache", "redis timeout")
        irrelevant = builder._calculate_relevance("lunch meeting agenda", "redis timeout")
        self.assertGreater(relevant, irrelevant)
        self.assertGreaterEqual(irrelevant, 0)
        self.assertLessEqual(relevant, 1)
        recent = builder._calculate_recency(NOW, now=NOW)
        old = builder._calculate_recency(NOW - timedelta(days=30), now=NOW)
        self.assertGreater(recent, old)
        self.assertTrue(0 <= old <= recent <= 1)
        newer = packet("newer", score=0.5)
        older = packet("older", score=0.5, timestamp=NOW - timedelta(days=30))
        selected = builder._select([older, newer], "question", available_tokens=100, now=NOW)
        self.assertEqual(selected[0], newer)

    def test_system_packet_precedes_higher_ranked_optional_evidence(self):
        builder = self.builder()
        system = packet("rules", "system_instruction", 0)
        evidence = packet("proof", "rag_result", 1)
        selected = builder._select([evidence, system], "question", available_tokens=20, now=NOW)
        self.assertEqual(selected[0], system)
        self.assertIn(evidence, selected)

    def test_large_optional_packet_does_not_block_smaller_following_packet(self):
        builder = self.builder(relevance_weight=1, recency_weight=0)
        too_big = packet("X" * 40, score=1)
        fits = packet("ok", score=0.8)
        selected = builder._select([too_big, fits], "question", available_tokens=5, now=NOW)
        self.assertNotIn(too_big, selected)
        self.assertIn(fits, selected)

    def test_explicit_half_relevance_is_not_treated_as_uncomputed(self):
        class DoNotRecalculate(ContextBuilder):
            def _calculate_relevance(self, content, query):
                raise AssertionError("显式 0.5 分不可重新计算")

        builder = DoNotRecalculate(ContextConfig(min_relevance=0.4), token_counter=CharacterCounter())
        known = packet("known score", score=0.5)
        selected = builder._select([known], "question", available_tokens=100, now=NOW)
        self.assertIn(known, selected)
        self.assertEqual(known.relevance_score, 0.5)


class StructureAndBudgetTests(unittest.TestCase):
    HEADINGS = ["[Role & Policies]", "[Task]", "[State]", "[Evidence]", "[Context]", "[Output]"]

    def test_structure_contains_all_six_sections_in_order(self):
        builder = ContextBuilder(token_counter=CharacterCounter())
        context = builder._structure([packet("keep-rule", "system_instruction", 1)], "keep-task")
        positions = [context.index(h) for h in self.HEADINGS]
        self.assertEqual(positions, sorted(positions))
        self.assertTrue(all(context.count(h) == 1 for h in self.HEADINGS))
        self.assertIn("keep-rule", context)
        self.assertIn("keep-task", context)
        self.assertIn(builder.OUTPUT_INSTRUCTION, context)

    def test_final_build_counts_headings_markers_and_optional_compression(self):
        counter = CharacterCounter()
        builder = ContextBuilder(ContextConfig(max_tokens=300, reserve_ratio=0.2), token_counter=counter)
        result = builder.build(
            "当前任务", system_instructions="保留这个系统约束",
            custom_packets=[packet("optional " * 100, score=0.8)], now=NOW,
        )
        self.assertIsInstance(result, ContextBuildResult)
        self.assertEqual(result.token_count, counter.count(result.context))
        self.assertLessEqual(result.token_count, builder.config.input_budget)
        for required in ["当前任务", "保留这个系统约束", builder.OUTPUT_INSTRUCTION, *self.HEADINGS]:
            self.assertIn(required, result.context)

    def test_headings_alone_can_exhaust_budget_even_when_body_is_short(self):
        builder = ContextBuilder(ContextConfig(max_tokens=30, reserve_ratio=0), token_counter=CharacterCounter())
        with self.assertRaises(ContextBudgetError):
            builder.build("q", system_instructions="s", now=NOW)

    def test_oversized_required_content_raises_instead_of_truncating_silently(self):
        builder = ContextBuilder(ContextConfig(max_tokens=180, reserve_ratio=0), token_counter=CharacterCounter())
        with self.assertRaises(ContextBudgetError):
            builder.build("task" * 100, system_instructions="policy", now=NOW)

    def test_oversized_system_policy_cannot_be_dropped_to_make_room(self):
        builder = ContextBuilder(ContextConfig(max_tokens=180, reserve_ratio=0), token_counter=CharacterCounter())
        with self.assertRaises(ContextBudgetError):
            builder.build("task", system_instructions="policy" * 100, now=NOW)

    def test_compression_keeps_required_sections_or_reports_budget_error(self):
        builder = ContextBuilder(token_counter=CharacterCounter())
        text = builder._structure([packet("policy", "system_instruction", 1), packet("filler " * 80)], "task")
        compressed = builder._compress(text, max_tokens=220)
        self.assertLessEqual(len(compressed), 220)
        for required in ["policy", "task", builder.OUTPUT_INSTRUCTION, *self.HEADINGS]:
            self.assertIn(required, compressed)
        with self.assertRaises(ContextBudgetError):
            builder._compress(text, max_tokens=10)
