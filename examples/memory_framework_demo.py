"""只检查 Memory 新框架的初始化，不读写业务文件，不调用任何模型。"""

from context_lab.memory import MemoryConfig, MemoryService
from context_lab.models import ScriptedChatModel
from context_lab.tools.memory import MemoryTool
from .common import ROOT


def main():
    path = ROOT / "runtime" / "memory-learning" / "memories.json"
    service = MemoryService(path, "student", config=MemoryConfig(), summary_model=ScriptedChatModel([]))
    tool = MemoryTool(path, "student", service=service)
    print("Memory 框架已装配：", type(tool.service).__name__)
    print("检索模式：", service.config.search_mode)
    print("三层：working / summary / archive；事实类型与层次独立。")
    print("M01—M08 已实现。阶段摘要演示可运行 python run.py example compaction-framework。")
    print("初始化不会创建业务目录、迁移数据、连接数据库或请求模型。")


if __name__ == "__main__":
    main()
