"""E05 后运行：文件读取示例，完全不执行 shell。"""

from context_lab.tools.terminal import TerminalTool
from .common import ROOT, run_learning_example


def main():
    terminal = TerminalTool(ROOT / "examples" / "sample_project", max_output_chars=1200)
    print(terminal.list_files())
    print(terminal.read_file("src/search.py", start_line=1, end_line=15))
    print(terminal.search("normalize"))


if __name__ == "__main__":
    run_learning_example(main)
