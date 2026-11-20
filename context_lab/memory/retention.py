"""M08：archive 记忆保留策略；候选筛选已完成，不在策略中写文件。"""

from datetime import datetime, timedelta, timezone

from .codec import MemoryCodec
from .config import MemoryConfig
from .types import MemoryCleanupPlan, MemorySnapshot, MemorySourceRef


class MemoryRetentionPolicy:
    def __init__(self, config: MemoryConfig | None = None):
        self.config = config if config is not None else MemoryConfig()

    def plan(
        self, snapshot: MemorySnapshot, user_id: str, now: datetime, limit: int | None = None,
    ) -> MemoryCleanupPlan:
        """计算保留截止点和可清理候选，返回计划，不修改输入或写文件。

        cutoff=now-timedelta(days=history_retention_days)，两类 archive 共用 90 天配置。
        只考虑当前用户、archive 层、valid_from 未晚于 now 的记录：
        - fact_key 非 None：必须有 valid_to，用 valid_to 作为保留期起点。
        - fact_key 为 None：有 valid_to 用 valid_to，否则用 updated_at。
        起点<=cutoff 才达到清理门槛；有明确有效期时不能回退到 updated_at。
        仍有效/未来生效的事实、working 和 summary 层不作为删除候选。
        metadata['pinned'] is True 的候选保留，protected 原因用 'pinned'。
        当前用户 fresh 摘要的 source_refs 所引用的候选保留，原因用 'summary_reference'；
        同时满足时优先标 pinned。stale 摘要不阻止清理；不检查其他用户的引用。
        两类候选合并，按各自保留期起点、id 升序，截取 limit
        （None 用配置批量值，0 为零候选），不是每类分别截取。
        candidates 保存 MemorySourceRef(id, version)，保留文件 revision；不改输入。
        fact 不因 updated_at 更新就延长保留期；无 valid_to 的普通经历按最近修改起算。
        查询/检索不更新 updated_at；不改输入、不写文件、不请求模型、不清向量缓存。
        """
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id 必须是非空字符串")
        if limit is not None and (type(limit) is not int or limit < 0):
            raise ValueError("limit 必须为 None 或非负整数")
        if not isinstance(now, datetime) or now.utcoffset() is None:
            raise ValueError("now 必须是带时区的 datetime")
        # 复用已有快照校验，不写文件、不改原对象。
        MemoryCodec().encode(snapshot)
        try:
            now = now.astimezone(timezone.utc)
            cutoff = now - timedelta(days=self.config.history_retention_days)
        except (ValueError, OverflowError) as error:
            raise ValueError("检查时间与保留期超出支持的时间范围") from error
        candidates = []
        summary_ids = set()
        protected = {}
        for record in snapshot.records:
            if record.user_id != user_id:
                continue
            if record.layer == 'summary' and record.summary_state == 'fresh':
                for ref in record.source_refs:
                    summary_ids.add(ref.memory_id)
        for record in snapshot.records:
            if record.user_id != user_id:
                continue
            if record.valid_from is not None and record.valid_from > now:
                continue
            end = None
            if record.layer == 'archive' and record.fact_key is None:
                if record.valid_to is None:
                    end = record.updated_at
                else:
                    end = record.valid_to
            elif record.layer == 'archive' and record.fact_key is not None:
                end = record.valid_to
            if end is not None and end <= cutoff:
                if record.metadata.get('pinned') is True:
                    protected[record.id] = 'pinned'
                elif record.id in summary_ids:
                    protected[record.id] = 'summary_reference'
                else:
                    candidates.append((end, record))
        candidates.sort(key=lambda item: (item[0], item[1].id))
        batch_limit = (
            self.config.history_cleanup_batch_size
            if limit is None
            else limit
        )
        candidates = [
            MemorySourceRef(record.id, record.version)
            for _, record in candidates[:batch_limit]
        ]

        return MemoryCleanupPlan(
            user_id=user_id, revision=snapshot.revision, checked_at=now,
            cutoff=cutoff, candidates=candidates, protected=protected,
        )
