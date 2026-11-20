"""演示记忆持久化、检索、事实修订和任务摘要。"""

import json
from datetime import timedelta

from context_lab.memory import MemoryService
from context_lab.models import ScriptedChatModel
from context_lab.types import utc_now
from .common import ROOT, run_learning_example


def main():
    now = utc_now()
    service = MemoryService(
        ROOT / "runtime" / "memory-learning" / "memories.json", "student",
        summary_model=ScriptedChatModel([json.dumps({
            "summary": "完成 Vector 返回类型兼容，本地测试验证了 Vector.to_list()。",
            "importance": 0.5,
        }, ensure_ascii=False)]),
    )
    print("业务快照：", service.repository.read_snapshot())  # M01
    service.add("检查 Vector 返回类型", layer="working", task_id="cache-fix")  # M02
    service.add("测试通过，支持 Vector.to_list()", layer="working", task_id="cache-fix")
    print("词法检索：", service.search("Vector", mode="lexical"))  # M03/M04
    old = service.add("项目使用 SQLite", memory_type="semantic", fact_key="project.database", valid_from=now)
    new = service.revise_fact(old.id, "项目使用 PostgreSQL", now + timedelta(days=1))  # M03
    print("新事实：", new.id)
    summary = service.finish_task("cache-fix")  # M06；离线预设文字只验收接线
    print("摘要：", summary.content)
    print("原始证据：", service.expand_summary(summary.id))
    print(service.prepare_context("修复向量读取", "next-task").render())


if __name__ == "__main__":
    run_learning_example(main)
