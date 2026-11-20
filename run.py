"""选择已安装的 Python 3.10+，不安装依赖或发送网络请求。

用法：python3 run.py overview / python3 run.py check scaffold
本启动器兼容 Python 3.9，避免系统旧解释器在项目类型注解上报错。
"""

from pathlib import Path
import os
import shutil
import subprocess
import sys


def main():
    if sys.version_info < (3, 10):
        for name in ("python3.13", "python3.12", "python3.11", "python3.10", "python3.14"):
            executable = shutil.which(name)
            if not executable:
                continue
            try:
                probe = subprocess.run(
                    [executable, "-c", "import sys; print(int(sys.version_info >= (3, 10)))"],
                    capture_output=True, text=True, timeout=5, check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                continue
            if probe.returncode == 0 and probe.stdout.strip() == "1":
                print("当前解释器版本较旧，使用已安装的 " + executable, flush=True)
                os.execv(executable, [executable, str(Path(__file__).resolve()), *sys.argv[1:]])
        print("需要 Python 3.10 或更新版本，未在 PATH 中找到可用解释器。", file=sys.stderr)
        print("请选择已有的新版解释器或创建相应环境，再运行此项目。", file=sys.stderr)
        return 1
    from context_lab.cli import main as cli_main
    return cli_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
