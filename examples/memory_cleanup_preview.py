"""M08 已完成的计划预览：只读，不申请实际删除。"""

from context_lab.memory import MemoryConfig, MemoryService
from .common import ROOT, run_learning_example


def main():
    service = MemoryService(ROOT / "runtime" / "memory-learning" / "memories.json", "student",
                            config=MemoryConfig(history_retention_days=90))
    plan = service.plan_history_cleanup()
    print("清理计划预览：", plan)


if __name__ == "__main__":
    run_learning_example(main)
