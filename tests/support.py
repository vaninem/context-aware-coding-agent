"""只提供隔离依赖和临时目录，不包含练习算法。"""

from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from context_lab.types import ContextBuildResult, Note


ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc)


class SandboxTestCase(unittest.TestCase):
    def setUp(self):
        # 临时数据始终在本项目 tests/ 下，测试结束后清理。
        tmp = TemporaryDirectory(prefix=".case-", dir=Path(__file__).parent)
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)


class CharacterCounter:
    """刻意采用字符度量，让 E06–E09 不依赖 E01 的答案。"""

    def __init__(self):
        self.calls = []

    def count(self, text):
        self.calls.append(text)
        return len(text)

    def truncate(self, text, max_tokens):
        if max_tokens < 0:
            raise ValueError("negative budget")
        return text[:max_tokens]


class FakeRetriever:
    def __init__(self, results=None, error=None):
        self.results = list(results or [])
        self.error = error
        self.calls = []

    def search(self, query, limit=5, **filters):
        self.calls.append((query, limit, filters))
        if self.error is not None:
            raise self.error
        return self.results[:limit]


class FakeBuilder:
    def __init__(self, error=None):
        self.calls = []
        self.error = error

    def build(self, user_query, **kwargs):
        self.calls.append((user_query, kwargs))
        if self.error is not None:
            raise self.error
        context = f"验收上下文：{user_query}"
        return ContextBuildResult(context, [], [], len(context))


class FakeNotes:
    def __init__(self):
        self.notes = []
        self.calls = []

    def create(self, title, content, note_type="general", tags=None):
        self.calls.append(("create", title, content))
        note = Note(f"note-{len(self.notes) + 1}", title, content, note_type, list(tags or []))
        self.notes.append(note)
        return note

    def search(self, query, limit=5, **filters):
        self.calls.append(("search", query))
        return self.notes[:limit]

    def list_notes(self, limit=20, **filters):
        self.calls.append(("list",))
        return self.notes[:limit]

    def summary(self):
        return {"total_notes": len(self.notes), "type_distribution": {}, "recent_notes": []}


class FakeTerminal:
    def __init__(self):
        self.calls = []

    def list_files(self, relative_path=".", pattern="*"):
        self.calls.append(("list", relative_path, pattern))
        return ["app.py", "README.md"]

    def read_file(self, relative_path, start_line=1, end_line=None):
        self.calls.append(("read", relative_path))
        return "def health_check():\n    return 'ok'\n"

    def search(self, query, relative_path=".", limit=20):
        self.calls.append(("search", query, relative_path))
        return [{"path": "app.py", "line": 1, "text": "def health_check():"}]
