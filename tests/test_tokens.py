"""E01：近似度量与前缀截断的自洽性。"""

import unittest

from context_lab.tokens import ApproxTokenCounter


class TokenTests(unittest.TestCase):
    def setUp(self):
        self.counter = ApproxTokenCounter()

    def test_chinese_is_counted_once_and_ascii_runs_as_one(self):
        self.assertEqual(self.counter.count("中文"), 2)
        self.assertEqual(self.counter.count("中文 Agent memory_v2 123"), 5)
        self.assertEqual(self.counter.count("a1_b"), 1)

    def test_punctuation_counts_but_whitespace_does_not(self):
        self.assertEqual(self.counter.count(" \t\n"), 0)
        self.assertEqual(self.counter.count("你好，world!"), 5)
        self.assertEqual(self.counter.count("a-b"), 3)

    def test_truncate_preserves_a_prefix_and_respects_its_own_count(self):
        original = "中文 Agent memory_v2，tail!"
        for budget in range(self.counter.count(original) + 1):
            with self.subTest(budget=budget):
                result = self.counter.truncate(original, budget)
                self.assertTrue(original.startswith(result))
                self.assertLessEqual(self.counter.count(result), budget)
                if budget > 0:
                    self.assertTrue(result)
        self.assertEqual(self.counter.truncate(original, 100), original)

    def test_empty_zero_and_negative_budgets(self):
        self.assertEqual(self.counter.truncate("", 3), "")
        self.assertEqual(self.counter.truncate("中文 hello", 0), "")
        with self.assertRaises(ValueError):
            self.counter.truncate("hello", -1)
