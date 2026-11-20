"""E05：只读取证、路径边界、行号和输出上限。"""

from context_lab.errors import WorkspaceViolation
from context_lab.tools.terminal import TerminalTool
from tests.support import SandboxTestCase


class TerminalTests(SandboxTestCase):
    def setUp(self):
        super().setUp()
        self.workspace = self.tmp / "workspace"
        self.workspace.mkdir()
        self.tool = TerminalTool(self.workspace)

    def test_list_and_one_based_inclusive_line_range(self):
        (self.workspace / "app.py").write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")
        (self.workspace / "notes.txt").write_text("text", encoding="utf-8")
        files = self.tool.list_files(pattern="*.py")
        self.assertTrue(any(p.endswith("app.py") for p in files))
        self.assertFalse(any(p.endswith("notes.txt") for p in files))
        text = self.tool.read_file("app.py", start_line=2, end_line=3)
        self.assertIn("two", text)
        self.assertIn("three", text)
        self.assertNotIn("one", text)
        self.assertNotIn("four", text)

    def test_list_files_recursion_and_budget_preserve_whole_sorted_paths(self):
        (self.workspace / "beta.py").touch()
        (self.workspace / "alpha.py").touch()
        (self.workspace / "src").mkdir()
        (self.workspace / "src" / "main.py").touch()

        self.assertEqual(self.tool.list_files(pattern="*.py"), ["alpha.py", "beta.py"])
        self.assertEqual(
            self.tool.list_files(pattern="**/*.py"), ["alpha.py", "beta.py", "src/main.py"]
        )
        self.assertEqual(self.tool.list_files(relative_path="src"), ["src/main.py"])
        limited_tool = TerminalTool(self.workspace, max_output_chars=8)
        self.assertEqual(limited_tool.list_files(pattern="**/*.py"), ["alpha.py"])

    def test_list_files_rejects_unsafe_patterns_before_descending(self):
        outside = self.tmp / "external"
        outside.mkdir()
        (self.workspace / "link").symlink_to(outside, target_is_directory=True)

        for pattern in ("../workspace/*.py", str(outside / "*.py"), "link/*.py", "l*/*.py"):
            with self.subTest(pattern=pattern), self.assertRaises(WorkspaceViolation):
                self.tool.list_files(pattern=pattern)

        (outside / "external.py").touch()
        (self.workspace / "external.py").symlink_to(outside / "external.py")
        with self.assertRaises(WorkspaceViolation):
            self.tool.list_files(pattern="*.py")

    def test_literal_search_returns_source_coordinates(self):
        (self.workspace / "sample.txt").write_text("axb\na.b\na.b again\n", encoding="utf-8")
        hits = self.tool.search("a.b", limit=1)
        self.assertEqual(len(hits), 1)
        self.assertTrue(hits[0]["path"].endswith("sample.txt"))
        self.assertEqual(hits[0]["line"], 2)
        self.assertIn("a.b", hits[0]["text"])

    def test_search_character_budget_counts_each_field_across_files(self):
        (self.workspace / "a.txt").write_text("匹配\n", encoding="utf-8")
        (self.workspace / "b.txt").write_text("匹配\n", encoding="utf-8")
        first_hit = {"path": "a.txt", "line": 1, "text": "匹配\n"}

        self.assertEqual(TerminalTool(self.workspace, 8).search("匹配"), [])
        self.assertEqual(TerminalTool(self.workspace, 9).search("匹配"), [first_hit])
        self.assertEqual(TerminalTool(self.workspace, 17).search("匹配"), [first_hit])
        self.assertEqual(len(TerminalTool(self.workspace, 18).search("匹配")), 2)

    def test_parent_traversal_is_rejected(self):
        (self.tmp / "outside.txt").write_text("outside", encoding="utf-8")
        with self.assertRaises(WorkspaceViolation):
            self.tool._resolve_path("../outside.txt")
        with self.assertRaises(WorkspaceViolation):
            self.tool.read_file("../outside.txt")

    def test_symlink_to_outside_workspace_is_rejected(self):
        outside = self.tmp / "outside.txt"
        outside.write_text("outside", encoding="utf-8")
        (self.workspace / "shortcut.txt").symlink_to(outside)
        with self.assertRaises(WorkspaceViolation):
            self.tool.read_file("shortcut.txt")

    def test_output_limit_includes_any_truncation_marker(self):
        (self.workspace / "long.txt").write_text("长文本" * 100, encoding="utf-8")
        result = TerminalTool(self.workspace, max_output_chars=17).read_file("long.txt")
        self.assertTrue(result)
        self.assertLessEqual(len(result), 17)

    def test_read_page_uses_complete_lines_and_exposes_the_next_line(self):
        (self.workspace / "pages.txt").write_bytes(b"abc\r\ndef\r\ngh")
        first = self.tool.read_file_page("pages.txt", max_tokens=8, count_tokens=len)
        self.assertEqual((first["text"], first["end_line"], first["next_start_line"]), ("abc\r\n", 1, 2))
        last = self.tool.read_file_page("pages.txt", start_line=first["next_start_line"], max_tokens=8, count_tokens=len)
        self.assertEqual((last["text"], last["end_line"], last["next_start_line"]), ("def\r\ngh", 3, None))

    def test_invalid_line_range_is_not_silently_reinterpreted(self):
        (self.workspace / "app.py").write_text("one\ntwo\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.tool.read_file("app.py", start_line=0)
        with self.assertRaises(ValueError):
            self.tool.read_file("app.py", start_line=2, end_line=1)

    def test_read_page_at_requested_end_needs_no_continuation(self):
        (self.workspace / "pages.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
        page = self.tool.read_file_page("pages.txt", end_line=2, max_tokens=40, count_tokens=len)
        self.assertEqual(page["text"], "one\ntwo\n")
        self.assertIsNone(page["next_start_line"])
