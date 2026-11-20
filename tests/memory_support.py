"""记忆练习的固定样例和假外部边界；不实现检索、时间过滤或持久化算法。"""

from copy import deepcopy
from datetime import datetime, timezone

from context_lab.memory import MemoryRecord

NOW = datetime(2026, 11, 1, 12, tzinfo=timezone.utc)


def record(memory_id="m1", content="redis timeout", user_id="alice", **changes):
    return MemoryRecord(**{
        "id": memory_id, "content": content, "user_id": user_id,
        "created_at": NOW, "updated_at": NOW, **changes,
    })


def wire_record(memory_id="m1", content="redis timeout", user_id="alice"):
    return {"id": memory_id, "content": content, "user_id": user_id,
            "memory_type": "episodic", "importance": 0.5,
            "created_at": NOW.isoformat(), "updated_at": NOW.isoformat(), "metadata": {},
            "layer": "archive", "valid_from": None, "valid_to": None, "task_id": None,
            "version": 1, "fact_key": None, "supersedes_id": None,
            "source_refs": [], "summary_state": "fresh"}


def document(*records, revision=0):
    return {"schema_version": 1, "revision": revision, "records": list(records), "index_events": []}


class RecordingMemoryIndex:
    """只记录桥接调用；返回人为指定的命中，不代表实际向量效果。"""
    def __init__(self, user_id="alice"):
        self.user_id = user_id
        self.namespace = "memory:" + user_id
        self.applied = []
        self.search_calls = []
        self.hits = []
        self.process_calls = []
        self.apply_error = None

    def apply(self, memory_id, item):
        if self.apply_error is not None:
            raise self.apply_error
        self.applied.append((memory_id, deepcopy(item)))

    def search(self, query, candidates, limit):
        self.search_calls.append((query, deepcopy(candidates), limit))
        return deepcopy(self.hits[:limit])

    def process_pending(self, limit=None):
        self.process_calls.append(limit)
        return 0
