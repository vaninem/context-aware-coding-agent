"""笔记更新的参数语义、持久化及保存失败恢复。"""

from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

from context_lab.tools.notes import NoteTool
from tests.support import SandboxTestCase


class NoteUpdateTests(SandboxTestCase):
    def setUp(self):
        super().setUp()
        self.tool = NoteTool(self.tmp / "notes")
        self.note = self.tool.create("原标题", "原正文", "general", ["old"])
        self.path = self.tool.workspace / f"{self.note.id}.md"
        self.old_data = self.path.read_bytes()
        self.old_index = self.tool.index_path.read_bytes()
        self.old_memory = deepcopy(self.tool.index)
        self.old_files = set(self.tool.workspace.iterdir())

    def assert_original_unchanged(self):
        self.assertEqual(self.path.read_bytes(), self.old_data)
        self.assertEqual(self.tool.index_path.read_bytes(), self.old_index)
        self.assertEqual(self.tool.index, self.old_memory)
        self.assertEqual(set(self.tool.workspace.iterdir()), self.old_files)
        self.assertEqual(NoteTool(self.tool.workspace).read(self.note.id), self.note)

    def test_update_persists_and_copies_tags(self):
        tags = ["Agent"]
        content = "  新正文\r\n\n---\n保留尾部空格  \n"
        updated = self.tool.update(
            self.note.id, title='新"标题"\n第二行', content=content,
            note_type="runbook", tags=tags,
        )
        tags.append("caller-change")
        self.assertEqual(updated.tags, ["Agent"])
        self.assertEqual(updated.id, self.note.id)
        self.assertEqual(updated.created_at, self.note.created_at)
        self.assertGreaterEqual(updated.updated_at, self.note.updated_at)
        restored = NoteTool(self.tool.workspace).read(self.note.id)
        self.assertEqual(restored, updated)
        self.assertEqual(restored.content, content)
        self.assertEqual(self.tool.index[self.note.id]["note_type"], "runbook")

    def test_none_preserves_fields_and_empty_tags_clears(self):
        unchanged = self.tool.update(self.note.id)
        self.assertEqual(unchanged.title, self.note.title)
        self.assertEqual(unchanged.content, self.note.content)
        self.assertEqual(unchanged.tags, ["old"])
        cleared = self.tool.update(self.note.id, tags=[])
        self.assertEqual(cleared.tags, [])
        self.assertEqual(NoteTool(self.tool.workspace).read(self.note.id), cleared)

    def test_invalid_values_do_not_change_files(self):
        cases = (
            {"title": " "}, {"title": 1}, {"content": ""},
            {"content": []}, {"note_type": "\n"}, {"note_type": False},
            {"tags": "Agent"}, {"tags": ("Agent",)},
            {"tags": [1]}, {"tags": [""]},
            {"title": "合法标题", "tags": [" "]},
        )
        for arguments in cases:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    self.tool.update(self.note.id, **arguments)
                self.assert_original_unchanged()

    def test_index_write_failure_restores_original(self):
        with patch.object(self.tool.store, "write", side_effect=OSError("模拟索引写入失败")):
            with self.assertRaisesRegex(OSError, "索引写入失败"):
                self.tool.update(self.note.id, title="新标题", content="新正文")
        self.assert_original_unchanged()

    def test_temporary_write_failure_preserves_original(self):
        with patch("context_lab.tools.notes.os.fsync", side_effect=OSError("模拟临时文件写入失败")):
            with self.assertRaisesRegex(OSError, "临时文件写入失败"):
                self.tool.update(self.note.id, title="新标题")
        self.assert_original_unchanged()

    def test_replace_failure_preserves_original(self):
        with patch.object(Path, "replace", side_effect=OSError("模拟替换失败")):
            with self.assertRaisesRegex(OSError, "替换失败"):
                self.tool.update(self.note.id, title="新标题")
        self.assert_original_unchanged()

    def test_clock_rollback_keeps_readable_timestamp(self):
        past = self.note.updated_at - timedelta(days=1)
        with patch("context_lab.tools.notes.utc_now", return_value=past):
            updated = self.tool.update(self.note.id, title="时钟回拨后的标题")
        self.assertGreaterEqual(updated.updated_at, self.note.updated_at)
        self.assertEqual(NoteTool(self.tool.workspace).read(self.note.id), updated)
