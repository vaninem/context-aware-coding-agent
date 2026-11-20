from pathlib import Path
from context_lab.errors import ExercisePending

ROOT = Path(__file__).resolve().parent.parent


def run_learning_example(function):
    try:
        function()
    except ExercisePending as exc:
        print(exc)
        print("请补齐对应练习后再执行；没有跳过 TODO，也没有生成虚构结果。")
        raise SystemExit(2)
