"""M01/M02：业务文件持久化；记忆和待同步事件在一次 JSON 替换中保存。"""

from contextlib import contextmanager
import fcntl
from pathlib import Path
from typing import Iterator

from ..storage.json_store import JsonFileStore
from .codec import MemoryCodec
from .errors import MemoryConflict
from .types import MemorySnapshot


class JsonMemoryRepository:
    def __init__(self, path: Path, user_id: str):
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id 必须为非空字符串")
        self.path = Path(path)
        self.user_id = user_id
        self.store = JsonFileStore(self.path)
        self.codec = MemoryCodec()
        self.lock_path = self.path.with_name(self.path.name + ".lock")

    @contextmanager
    def locked(self, *, index_sync: bool = False) -> Iterator[None]:
        """分别为业务提交和索引同步提供文件锁。"""
        lock_path = self.lock_path.with_suffix(".index-sync.lock") if index_sync else self.lock_path
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def read_snapshot(self) -> MemorySnapshot:
        """读取文件、恢复对象，再返回当前用户的独立快照。

        revision 是文件级版本；records/index_events 都按 self.user_id 过滤。
        每次 decode 都生成新对象，不缓存读取结果；缺失文件使用空快照。
        文件内容损坏或读失败继续抛出，不用空数据覆盖错误，也不改写文件。
        """
        payload = self.store.read(default=self.codec.empty_document())
        snapshot = self.codec.decode(payload)
        user_records = []
        user_events = []
        for record in snapshot.records:
            if record.user_id == self.user_id:
                user_records.append(record)
        for event in snapshot.index_events:
            if event.user_id == self.user_id:
                user_events.append(event)
        return MemorySnapshot(
            revision=snapshot.revision, records=user_records, index_events=user_events,
        )

    def commit(self, snapshot: MemorySnapshot) -> MemorySnapshot:
        """用用户完整新快照替换该用户分区，保留其他用户。

        在 with self.locked() 内读最新文件、核对 snapshot.revision、合并、写入。
        版本不一致抛 MemoryConflict；记录和事件均拒绝混入其他用户。
        全局 revision 加一，单次 JsonFileStore.write 原子保存业务记录和 outbox。
        不修改传入快照；写成功后才返回当前用户的新副本（含新文件版本），
        不能返回其他用户数据；写失败不能更新调用方内存或返回成功。
        始终写入统一字典格式，不迁移旧列表文件。
        """
        # 复用 Codec 校验，得到独立且时间统一的副本，不修改调用方对象。
        incoming = self.codec.decode(self.codec.encode(snapshot))
        if any(record.user_id != self.user_id for record in incoming.records) or any(
            event.user_id != self.user_id for event in incoming.index_events
        ):
            raise ValueError("提交的记忆和索引事件必须全部属于当前用户")

        with self.locked():
            payload = self.store.read(default=self.codec.empty_document())
            old_snapshot = self.codec.decode(payload)
            if old_snapshot.revision != incoming.revision:
                raise MemoryConflict(
                    f"快照版本 {incoming.revision} 与文件版本 {old_snapshot.revision} 不一致，请重新读取后重试"
                )
            new_records = []
            new_events = []
            for record in old_snapshot.records:
                if record.user_id != self.user_id:
                    new_records.append(record)
            new_records.extend(incoming.records)
            for event in old_snapshot.index_events:
                if event.user_id != self.user_id:
                    new_events.append(event)
            new_events.extend(incoming.index_events)
            latest_snapshot = MemorySnapshot(old_snapshot.revision + 1, new_records, new_events)
            latest_dict = self.codec.encode(latest_snapshot)
            # 返回值先准备好，写成功后只返回；incoming 与原输入已经互相独立。
            result = MemorySnapshot(
                revision=latest_snapshot.revision,
                records=incoming.records,
                index_events=incoming.index_events,
            )
            self.store.write(latest_dict)
            return result

    def acknowledge(self, event_ids: list[str]) -> None:
        """从当前用户的待办中移除指定事件，保留业务记录和其他事件。

        已确认的 ID 可以重复提交。版本冲突时重新读取最新快照，最多尝试三次。
        """
        if not isinstance(event_ids, list) or any(
            not isinstance(event_id, str) or not event_id.strip() for event_id in event_ids
        ):
            raise ValueError("event_ids 必须是由非空字符串组成的列表")
        if not event_ids:
            return

        event_id_set = set(event_ids)
        for attempt in range(3):
            snapshot = self.read_snapshot()
            remaining_events = [event for event in snapshot.index_events if event.id not in event_id_set]
            if len(remaining_events) == len(snapshot.index_events):
                return
            snapshot.index_events = remaining_events
            try:
                self.commit(snapshot)
            except MemoryConflict:
                if attempt == 2:
                    raise
                continue
            return
