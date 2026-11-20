"""显式启动每小时维护循环，会实际按策略清理；Ctrl+C 停止。"""

from threading import Event

from context_lab.memory import MemoryConfig, MemoryService
from .common import ROOT, run_learning_example


def main():
    service = MemoryService(
        ROOT / "runtime" / "memory-learning" / "memories.json", "student",
        config=MemoryConfig(history_retention_days=90, auto_cleanup_history=True),
    )
    stop = Event()
    print("准备维护 archive：fact 从失效起、普通经历无有效期时从最近修改起保留 90 天。")
    print("每小时检查，两类合计每批最多 100 条。")
    try:
        service.history_cleanup_worker.run(stop)
    except KeyboardInterrupt:
        stop.set()
        print("archive 记忆维护已停止。")


if __name__ == "__main__":
    run_learning_example(main)
