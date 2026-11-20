"""M08 已完成的定时驱动；默认关闭，显式启动后才按策略运行。"""

from datetime import datetime
from threading import Event, Lock
from typing import TYPE_CHECKING

from .types import MemoryCleanupReport

if TYPE_CHECKING:
    from .service import MemoryService


class MemoryCleanupWorker:
    def __init__(self, service: "MemoryService"):
        self.service = service
        self._last_success_at: datetime | None = None
        self._lock = Lock()

    @property
    def last_success_at(self) -> datetime | None:
        return self._last_success_at

    def tick(self) -> MemoryCleanupReport | None:
        """已提供：启用且到期才调用一次清理；异常继续抛出，失败不推进时间。"""
        with self._lock:
            if not self.service.config.auto_cleanup_history:
                return None
            now = self.service.clock()
            if not isinstance(now, datetime) or now.utcoffset() is None:
                raise ValueError("清理时钟必须返回带时区的 datetime")
            if self._last_success_at is not None:
                elapsed = (now - self._last_success_at).total_seconds()
                if elapsed < self.service.config.history_cleanup_interval_seconds:
                    return None
            report = self.service.cleanup_history(
                dry_run=False, limit=self.service.config.history_cleanup_batch_size,
            )
            if not isinstance(report, MemoryCleanupReport) or report.dry_run:
                raise ValueError("自动清理必须返回实际执行的 MemoryCleanupReport")
            self.service.last_cleanup_report = report
            self._last_success_at = now
            return report

    def run(self, stop_event: Event) -> None:
        """已提供：独立维护循环；停止事件可中断等待，错误交给调用方处理。"""
        while not stop_event.is_set():
            self.tick()
            stop_event.wait(self.service.config.history_cleanup_interval_seconds)
