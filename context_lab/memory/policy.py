"""M03：先确定记忆在本次查询中是否有资格，再计算相关性。"""

from copy import deepcopy
from datetime import datetime

from .types import MemoryQuery, MemoryRecord


class MemoryPolicy:
    def select(
        self, records: list[MemoryRecord], user_id: str, query: MemoryQuery, now: datetime,
    ) -> list[MemoryRecord]:
        """用户→有效时间→层次/任务/类型/重要性过滤；返回副本。

        t=query.as_of 或 now；有效条件为 (from is None or from<=t) 且
        (to is None or t<to)，区间右侧不包含。不能用 updated_at 代替有效期。
        过期事实可在旧 as_of 查询命中；创建时间晚于 as_of 不自动排除回填事实。
        不接受调用参数替换服务 user_id；重要性 0 是合法边界。
        stale 摘要不进入正常检索；未来才生效的事实不在当前结果里。
        """
        if not isinstance(records, list) or not isinstance(query, MemoryQuery):
            raise ValueError("records 必须是记忆列表，query 必须是 MemoryQuery")
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id 必须是非空字符串")
        for name, value in (("now", now), ("as_of", query.as_of)):
            if name == "as_of" and value is None:
                continue
            if not isinstance(value, datetime) or value.utcoffset() is None:
                raise ValueError(f"{name} 必须是带时区的 datetime")
        if query.layers is not None and (
            not isinstance(query.layers, tuple)
            or any(layer not in ("working", "summary", "archive") for layer in query.layers)
        ):
            raise ValueError("layers 必须为 None 或由 working/summary/archive 组成的元组")
        for name in ("task_id", "memory_type"):
            value = getattr(query, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} 必须为 None 或非空字符串")
        for name in ("min_importance", "max_importance"):
            value = getattr(query, name)
            if value is not None and (type(value) not in (int, float) or not 0 <= value <= 1):
                raise ValueError(f"{name} 必须为 None 或 0 到 1 之间的有限数值")
        if (query.min_importance is not None and query.max_importance is not None
                and query.min_importance > query.max_importance):
            raise ValueError("min_importance 不能大于 max_importance")

        results = []
        t = now if query.as_of is None else query.as_of
        for record in records:
            if not isinstance(record, MemoryRecord):
                raise ValueError("records 中每一项必须是 MemoryRecord")
            if record.user_id != user_id:
                continue
            for name in ("valid_from", "valid_to"):
                value = getattr(record, name)
                if value is not None and (not isinstance(value, datetime) or value.utcoffset() is None):
                    raise ValueError(f"记忆 {record.id} 的 {name} 必须为 None 或带时区的 datetime")
            if record.valid_from is not None and record.valid_to is not None and record.valid_from >= record.valid_to:
                raise ValueError(f"记忆 {record.id} 的有效区间必须满足 valid_from < valid_to")
            if type(record.importance) not in (int, float) or not 0 <= record.importance <= 1:
                raise ValueError(f"记忆 {record.id} 的 importance 必须是 0 到 1 之间的有限数值")
            if record.valid_from is not None and record.valid_from > t:
                continue
            if record.valid_to is not None and record.valid_to <= t:
                continue
            if record.layer == "summary" and record.summary_state == "stale":
                continue
            if query.layers is not None and record.layer not in query.layers:
                continue
            if query.task_id is not None and record.task_id != query.task_id:
                continue
            if query.memory_type is not None and record.memory_type != query.memory_type:
                continue
            if query.min_importance is not None and record.importance < query.min_importance:
                continue
            if query.max_importance is not None and record.importance > query.max_importance:
                continue
            results.append(deepcopy(record))
        return results

