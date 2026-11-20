"""从冻结的 Python 项目中推断测试入口，仓库文件只作为识别线索。"""

import ast
import os
import re
from pathlib import Path
from typing import TypedDict

from .sandbox import SNAPSHOT_EXCLUDES


class TestPlan(TypedDict):
    command: list[str]
    source: str


PYTEST = ("python", "-m", "pytest", "-q")
PYTEST_SECTIONS = {
    "pyproject.toml": r"\[tool\.pytest(?:\.ini_options)?\]",
    "setup.cfg": r"\[tool:pytest\]",
    "tox.ini": r"\[pytest\]",
}
TEST_BASES = {"TestCase", "IsolatedAsyncioTestCase"}
FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
EXCLUDED_TEST_DIRS = SNAPSHOT_EXCLUDES | {"build", "dist", "runtime", "site-packages"}


def _test_style(path: Path) -> str | None:
    try:
        tree = ast.parse(path.read_bytes(), filename=path.name)
    except (SyntaxError, ValueError):
        if path.name.endswith("_test.py"):
            return "pytest"
        return "unittest" if path.name.startswith("test_") else None

    unittest = False
    for node in tree.body:
        if isinstance(node, FUNCTIONS) and node.name.startswith("test"):
            return "pytest"
        if isinstance(node, ast.ClassDef):
            bases = {base.id for base in node.bases if isinstance(base, ast.Name)}
            bases.update(base.attr for base in node.bases if isinstance(base, ast.Attribute))
            if bases & TEST_BASES:
                unittest = True
            elif node.name.startswith("Test") and any(
                isinstance(method, FUNCTIONS) and method.name.startswith("test") for method in node.body
            ):
                return "pytest"
    if unittest:
        return "pytest" if path.name.endswith("_test.py") else "unittest"
    return None


def _find_tests(root: Path, *, standard_dirs: bool) -> list[tuple[Path, str]]:
    tests = []
    for directory, subdirs, files in os.walk(root):
        parent = Path(directory)
        if parent == root and standard_dirs:
            subdirs[:] = sorted(set(subdirs) & {"test", "tests"})
        else:
            subdirs[:] = sorted(name for name in subdirs if not name.startswith(".") and name not in EXCLUDED_TEST_DIRS)
        for name in sorted(files):
            if not (name.startswith("test") and name.endswith(".py") or name.endswith("_test.py")):
                continue
            path = parent / name
            if not path.is_symlink() and path.is_file() and (style := _test_style(path)):
                tests.append((path.relative_to(root), style))
    return tests


def discover_tests(root: Path) -> TestPlan:
    """生成从项目根目录运行的命令，或返回未发现入口的原因。"""
    for name in ("pytest.ini", ".pytest.ini", "pytest.toml", ".pytest.toml"):
        if (root / name).is_file():
            return {"command": list(PYTEST), "source": name}
    for name, section in PYTEST_SECTIONS.items():
        path = root / name
        if path.is_file() and re.search(rf"^\s*{section}\s*(?:#.*)?$", path.read_text(encoding="utf-8", errors="replace"), re.M):
            return {"command": list(PYTEST), "source": name}

    tests = _find_tests(root, standard_dirs=True) or _find_tests(root, standard_dirs=False)
    if not tests:
        return {"command": [], "source": "未发现 Python 测试入口"}
    start = Path(os.path.commonpath([str(path.parent) for path, _ in tests]))
    # unittest 在起始目录内继续递归时要求中间目录是包。pytest 可收集普通目录。
    for path, style in tests:
        parents = path.relative_to(start).parents
        if style == "pytest" or any(parent != Path(".") and not (root / start / parent / "__init__.py").is_file()
                                    for parent in parents):
            return {"command": list(PYTEST), "source": path.as_posix()}
    location = start.as_posix()
    command_start = f"./{location}" if location.startswith("-") else location
    return {"command": ["python", "-m", "unittest", "discover", "-s", command_start], "source": location}
