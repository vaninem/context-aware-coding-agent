"""E02：笔记 CRUD、文件持久化和跨进程读取。"""

import json
import subprocess
import sys

from context_lab.tools.notes import NoteTool
from context_lab.types import Note
from tests.support import ROOT, SandboxTestCase


class NoteTests(SandboxTestCase):
    def make_tool(self):
        return NoteTool(self.tmp / "notes")

    def test_create_and_update_keep_identity_and_metadata(self):
        tool = self.make_tool()
        note = tool.create("缓存排查", "先检查命中率", "runbook", ["cache"])
        self.assertIsInstance(note, Note)
        updated = tool.update(note.id, title="缓存处理", content="再检查超时", tags=["timeout"])
        self.assertEqual(updated.id, note.id)
        self.assertEqual(updated.created_at, note.created_at)
        self.assertEqual(updated.note_type, "runbook")
        self.assertEqual(updated.tags, ["timeout"])
        self.assertEqual(tool.read(note.id).content, "再检查超时")
        self.assertIsNotNone(updated.updated_at.utcoffset())
        self.assertGreaterEqual(updated.updated_at, note.created_at)
        cleared = tool.update(note.id, tags=[])
        self.assertEqual(cleared.tags, [])
        self.assertEqual(cleared.title, "缓存处理")

    def test_markdown_front_matter_and_json_index_are_written(self):
        tool = self.make_tool()
        note = tool.create("特殊标题：缓存", "原始正文", "runbook", ["cache"])
        files = list(tool.workspace.rglob("*.md"))
        self.assertEqual(len(files), 1)
        saved = files[0].read_text(encoding="utf-8")
        self.assertTrue(saved.startswith("---\n"))
        self.assertGreaterEqual(saved.count("---"), 2)
        self.assertIn(note.id, saved)
        self.assertIn("原始正文", saved)
        with tool.index_path.open(encoding="utf-8") as handle:
            self.assertIsNotNone(json.load(handle))

    def test_search_filters_limit_and_summary(self):
        tool = self.make_tool()
        target = tool.create("缓存超时", "redis timeout", "runbook", ["cache"])
        tool.create("会议记录", "redis timeout", "meeting", ["team"])
        tool.create("磁盘手册", "disk full", "runbook", ["disk"])
        hits = tool.search("redis", limit=1, note_type="runbook", tags=["cache"])
        self.assertEqual([n.id for n in hits], [target.id])
        self.assertLessEqual(len(tool.list_notes(limit=1)), 1)
        self.assertTrue(all(n.note_type == "runbook" for n in tool.list_notes(note_type="runbook")))
        summary = tool.summary()
        self.assertEqual(summary["total_notes"], 3)
        self.assertEqual(summary["type_distribution"]["runbook"], 2)
        self.assertIsInstance(summary["recent_notes"], list)

    def test_another_python_process_can_read_a_saved_note(self):
        tool = self.make_tool()
        note = tool.create("重启后的笔记", "应从磁盘恢复", tags=["persist"])
        script = (
            "import json,sys; from pathlib import Path; "
            "from context_lab.tools.notes import NoteTool; "
            "n=NoteTool(Path(sys.argv[1])).read(sys.argv[2]); "
            "print(json.dumps({'id':n.id,'content':n.content,'tags':n.tags},ensure_ascii=False))"
        )
        result = subprocess.run(
            [sys.executable, "-c", script, str(tool.workspace), note.id],
            cwd=ROOT, capture_output=True, text=True, timeout=10, check=True,
        )
        data = json.loads(result.stdout)
        self.assertEqual(data, {"id": note.id, "content": "应从磁盘恢复", "tags": ["persist"]})

    def test_delete_is_persisted(self):
        tool = self.make_tool()
        note = tool.create("待删除", "old")
        self.assertTrue(tool.delete(note.id))
        self.assertFalse(tool.delete(note.id))
        self.assertNotIn(note.id, [n.id for n in NoteTool(tool.workspace).list_notes()])
