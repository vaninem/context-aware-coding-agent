"""M03 筛选检查：时间有效性、组合条件、用户边界和只读副本。"""

from copy import deepcopy
from datetime import timedelta, timezone
import unittest

from context_lab.memory import MemoryPolicy, MemoryQuery, MemorySourceRef
from tests.memory_support import NOW, record


class MemoryPolicyTests(unittest.TestCase):
    def test_uses_supplied_now_or_as_of_and_keeps_half_open_time_boundaries(self):
        past = record("past", valid_from=NOW - timedelta(hours=1), valid_to=NOW)
        current = record("current", valid_from=NOW, valid_to=NOW + timedelta(hours=1))
        future = record("future", valid_from=NOW + timedelta(hours=1))
        undated = record("undated")
        records = [future, undated, past, current]
        policy = MemoryPolicy()
        self.assertEqual(policy.select(records, "alice", MemoryQuery(), NOW), [undated, current])
        self.assertEqual(policy.select(records, "alice", MemoryQuery(as_of=NOW - timedelta(minutes=30)), NOW),
                         [undated, past])
        self.assertEqual(policy.select(records, "alice", MemoryQuery(), NOW + timedelta(hours=1)),
                         [future, undated])
        same_instant = NOW.astimezone(timezone(timedelta(hours=8)))
        self.assertEqual(policy.select(records, "alice", MemoryQuery(as_of=same_instant), NOW), [undated, current])

    def test_optional_filters_combine_and_zero_and_empty_layers_are_not_ignored(self):
        a = record("a", importance=0)
        b = record("b", layer="working", task_id="t1", memory_type="semantic", importance=0.5)
        c = record("c", task_id="t2", memory_type="semantic", importance=1)
        d = record("d", layer="summary", task_id="t1", importance=0.7)
        stale = record("stale", layer="summary", summary_state="stale", importance=1)
        records = [a, b, c, d, stale]
        policy = MemoryPolicy()
        self.assertEqual(policy.select(records, "alice", MemoryQuery(), NOW), [a, b, c, d])
        self.assertEqual(policy.select(records, "alice", MemoryQuery(layers=()), NOW), [])
        self.assertEqual(policy.select(records, "alice", MemoryQuery(layers=("archive",)), NOW), [a, c])
        self.assertEqual(policy.select(records, "alice", MemoryQuery(task_id="t1"), NOW), [b, d])
        self.assertEqual(policy.select(records, "alice", MemoryQuery(max_importance=0), NOW), [a])
        self.assertEqual(policy.select(records, "alice", MemoryQuery(min_importance=1), NOW), [c])
        query = MemoryQuery(layers=("archive", "working"), task_id="t1", memory_type="semantic",
                            min_importance=0.5, max_importance=0.5)
        self.assertEqual(policy.select(records, "alice", query, NOW), [b])

    def test_results_keep_original_order_and_do_not_share_nested_objects_with_input(self):
        a = record("a", layer="summary", metadata={"tags": ["original"]},
                   source_refs=[MemorySourceRef("source", 1)])
        b = record("b")
        records = [b, a]
        before = deepcopy(records)
        selected = MemoryPolicy().select(records, "alice", MemoryQuery(), NOW)
        self.assertEqual([item.id for item in selected], ["b", "a"])
        self.assertEqual(records, before)
        selected[-1].metadata["tags"].append("local change")
        selected[-1].source_refs[0].version = 2
        self.assertEqual(records, before)

    def test_invalid_query_shapes_and_filter_values_raise_clear_errors(self):
        base = {"records": [], "user_id": "alice", "query": MemoryQuery(), "now": NOW}
        cases = [{"records": ()}, {"records": [{}]}, {"user_id": None}, {"user_id": " "},
                 {"query": None}, {"query": {}}, {"query": MemoryQuery(layers=[])},
                 {"query": MemoryQuery(layers=("other",))}, {"query": MemoryQuery(task_id=" ")},
                 {"query": MemoryQuery(memory_type=3)}]
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                MemoryPolicy().select(**{**base, **changes})

    def test_naive_or_invalid_times_and_reversed_intervals_are_rejected(self):
        policy = MemoryPolicy()
        for value in (None, "now", NOW.replace(tzinfo=None)):
            with self.subTest(now=value), self.assertRaises(ValueError):
                policy.select([], "alice", MemoryQuery(), value)
        for value in ("yesterday", NOW.replace(tzinfo=None)):
            with self.subTest(as_of=value), self.assertRaises(ValueError):
                policy.select([], "alice", MemoryQuery(as_of=value), NOW)
        for key in ("valid_from", "valid_to"):
            for value in ("yesterday", NOW.replace(tzinfo=None)):
                with self.subTest(field=key, value=value), self.assertRaises(ValueError):
                    policy.select([record(**{key: value})], "alice", MemoryQuery(), NOW)
        for end in (NOW, NOW - timedelta(seconds=1)):
            with self.subTest(end=end), self.assertRaises(ValueError):
                policy.select([record(valid_from=NOW, valid_to=end)], "alice", MemoryQuery(), NOW)

    def test_invalid_importance_bounds_and_record_scores_are_rejected(self):
        policy = MemoryPolicy()
        for key in ("min_importance", "max_importance"):
            for value in (True, -0.1, 1.1, float("nan"), float("inf"), "0.5"):
                with self.subTest(field=key, value=value), self.assertRaises(ValueError):
                    policy.select([], "alice", MemoryQuery(**{key: value}), NOW)
        with self.assertRaises(ValueError):
            policy.select([], "alice", MemoryQuery(min_importance=0.8, max_importance=0.2), NOW)
        for value in (True, -1, float("nan"), "0.5"):
            with self.subTest(importance=value), self.assertRaises(ValueError):
                policy.select([record(importance=value)], "alice", MemoryQuery(), NOW)

    def test_foreign_records_are_excluded_before_evaluating_their_filter_fields(self):
        foreign = record("foreign", user_id="bob", valid_from=NOW.replace(tzinfo=None), importance="unknown")
        own = record("own")
        self.assertEqual(MemoryPolicy().select([foreign, own], "alice", MemoryQuery(), NOW), [own])
