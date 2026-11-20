"""E01、E06-08 后运行：python -m examples.context_builder_demo。"""

from context_lab.config import ContextConfig
from context_lab.context import ContextBuilder
from context_lab.types import ContextPacket, Message
from .common import run_learning_example


def main():
    builder = ContextBuilder(config=ContextConfig(max_tokens=500, reserve_ratio=0.2))
    result = builder.build(
        user_query="下一步应当检查哪个问题？",
        system_instructions="仅根据已读取材料回答，不执行代码变更。",
        conversation_history=[Message("user", "我正在学习文档检索。")],
        custom_packets=[ContextPacket(
            content="任务阻塞：带首尾空白的查询没有规范化。",
            relevance_score=0.9,
            metadata={"type": "note", "note_type": "blocker", "note_id": "example"},
        )],
    )
    print(result.context)
    print("最终近似 token：", result.token_count)


if __name__ == "__main__":
    run_learning_example(main)
