"""笔记读取的持久化、格式校验与路径边界。"""

import json
from datetime import timedelta

from context_lab.tools.notes import NoteTool
from tests.support import SandboxTestCase


class NoteReadTests(SandboxTestCase):
    def setUp(self):
        super().setUp()
        self.tool = NoteTool(self.tmp / "notes")
        self.note = self.tool.create('含"引号"\n标题', "  正文\r\n\n---\n尾部\n ", tags=["Agent"])
        self.path = self.tool.workspace / f"{self.note.id}.md"
        self.original = self.path.read_bytes()
        self.entry = dict(self.tool.index[self.note.id])

    def write_metadata(self, metadata):
        header = "\n".join(f"{key}: {json.dumps(value, ensure_ascii=False)}" for key, value in metadata.items())
        self.path.write_text(f"---\n{header}\n---\n\n{self.note.content}", encoding="utf-8")

    def test_new_instance_preserves_note_and_body(self):
        restored = NoteTool(self.tool.workspace).read(self.note.id)
        self.assertEqual(restored, self.note)
        self.assertEqual(restored.created_at.utcoffset(), timedelta(0))

    def test_invalid_ids_are_rejected(self):
        for note_id in (None, [], "../escape", "note_abc", self.note.id + "\n"):
            with self.subTest(note_id=note_id), self.assertRaises(ValueError):
                self.tool.read(note_id)

    def test_missing_duplicate_and_invalid_front_matter(self):
        for text in ("正文", "---\n", '---\nid: "abc"\n正文', '---\nid: "a"\nid: "b"\n---\n\n正文', "---\nid: invalid\n---\n\n正文"):
            with self.subTest(text=text):
                self.path.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.tool.read(self.note.id)

    def test_metadata_fields_types_and_times_are_checked(self):
        baseline = {key: value for key, value in self.entry.items() if key != "file_path"}
        cases = [
            {key: value for key, value in baseline.items() if key != "title"},
            {**baseline, "content": "意外字段"},
            {**baseline, "id": "wrong"},
            {**baseline, "title": " "},
            {**baseline, "note_type": 1},
            {**baseline, "tags": "Agent"},
            {**baseline, "tags": [1]},
            {**baseline, "created_at": 123},
            {**baseline, "created_at": "bad-date"},
            {**baseline, "created_at": "2026-01-01T00:00:00"},
            {**baseline, "updated_at": "2000-01-01T00:00:00+00:00"},
        ]
        for metadata in cases:
            with self.subTest(metadata=metadata):
                # 同时修改索引，确保不仅是“不一致”检查拦住损坏数据。
                self.tool.store.write({self.note.id: {**metadata, "file_path": self.path.name}})
                self.write_metadata(metadata)
                with self.assertRaises(ValueError):
                    self.tool.read(self.note.id)

    def test_index_mismatch_and_paths_are_rejected(self):
        for change in ({"title": "索引中的其他标题"}, {"file_path": "../outside.md"}, {"id": "wrong"}):
            with self.subTest(change=change):
                self.tool.store.write({self.note.id: {**self.entry, **change}})
                with self.assertRaises(ValueError):
                    self.tool.read(self.note.id)

        self.tool.store.write({self.note.id: self.entry})
        outside = self.tmp / "outside.md"
        outside.write_bytes(self.original)
        self.path.unlink()
        self.path.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "超出工作区"):
            self.tool.read(self.note.id)
