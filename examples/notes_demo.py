"""E02 后运行：创建、读取、更新并重建工具对象，再读同一笔记。"""

from context_lab.tools.notes import NoteTool
from .common import ROOT, run_learning_example


def main():
    workspace = ROOT / "runtime" / "notes-demo"
    notes = NoteTool(workspace)
    note = notes.create(
        title="检索功能练习", content="已完成：阅读接口。下一步：实现笔记存取。",
        note_type="task_state", tags=["learning"],
    )
    print(notes.read(note.id))
    notes.update(note.id, content="已完成：笔记生命周期练习。", note_type="conclusion")
    reloaded = NoteTool(workspace)
    print(reloaded.read(note.id))
    print(reloaded.summary())


if __name__ == "__main__":
    run_learning_example(main)
