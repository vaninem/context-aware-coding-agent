"""M01—M04 和 E04 后运行：新版用户记忆与外部知识来源。"""

from context_lab.tools.memory import MemoryTool
from context_lab.tools.rag import RAGTool
from .common import ROOT, run_learning_example


def main():
    memory = MemoryTool(ROOT / "runtime" / "memory-demo.json", user_id="student")
    memory.add("我正在学习 chunk overlap 的原理。", memory_type="semantic", importance=0.8)
    print("记忆：", memory.search("chunk overlap"))
    rag = RAGTool(ROOT / "examples" / "knowledge_base", chunk_size=120, chunk_overlap=30)
    rag.add_document(ROOT / "examples" / "knowledge_base" / "chunking.md")
    print("知识片段：", rag.search("overlap"))


if __name__ == "__main__":
    run_learning_example(main)
