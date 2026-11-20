"""E01-09 后运行：把工具、笔记、上下文和有限测试模型连起来。"""

from context_lab.cli import make_demo_agent
from .common import run_learning_example


def main():
    agent = make_demo_agent()
    print(agent.explore().answer)
    print(agent.analyze_issue("为什么查询带额外空格时会漏匹配？").answer)
    print(agent.plan_task("为查询规范化制定可验证的修改计划。").answer)
    print(agent.get_stats())


if __name__ == "__main__":
    run_learning_example(main)
