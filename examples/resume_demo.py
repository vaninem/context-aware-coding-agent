"""E09 后运行：保存业务状态，在新助手对象里恢复。"""

from context_lab.cli import make_demo_agent
from context_lab.types import Message
from .common import run_learning_example


def main():
    first = make_demo_agent()
    first.history.append(Message("user", "先检查查询规范化，再检查重排。"))
    saved = first.save_checkpoint()
    second = make_demo_agent()
    state = second.load_checkpoint()
    print("文件：", saved)
    print("恢复结果：", state)
    print("历史消息：", second.history)
    print("这只恢复本地业务状态，不恢复操作系统进程。")


if __name__ == "__main__":
    run_learning_example(main)
