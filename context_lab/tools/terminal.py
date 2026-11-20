"""E05：受工作区约束的只读文件工具；不会执行任意 shell 命令。"""

from collections.abc import Callable
from itertools import islice
from pathlib import Path
from stat import S_ISDIR, S_ISREG
from typing import Any

from ..errors import WorkspaceViolation
from ..types import ToolResult
from .base import BaseTool


class TerminalTool(BaseTool):
    """学习版 Terminal 仅包含文件列表、逐行读取和字面关键词搜索。

    工具名称沿用章节命名，不代表系统终端权限。所有文件访问必须先经过
    _resolve_path；越界、外部符号链接和输出长度限制属于 E05 的核心作业。
    """

    def __init__(self, workspace: Path, max_output_chars: int = 8000):
        """保存解析后的工作区和输出字符上限；不读取文件、不运行子进程。"""
        super().__init__("terminal", "工作区内只读文件列表、读取与字面搜索")
        if type(max_output_chars) is not int or max_output_chars <= 0:
            raise ValueError("max_output_chars 必须为正整数")
        self.workspace = Path(workspace).resolve()
        self.max_output_chars = max_output_chars

    def run(self, parameters: dict[str, Any]) -> ToolResult:
        """分发 list_files/read_file/search；任何 command/shell 字段均不在接口内。"""
        if not isinstance(parameters, dict):
            raise ValueError("工具参数必须为字典")
        action = parameters.get("action")
        routes = {
            "list_files": (self.list_files, {"relative_path", "pattern"}, set()),
            "read_file": (self.read_file, {"relative_path", "start_line", "end_line"}, {"relative_path"}),
            "search": (self.search, {"query", "relative_path", "limit"}, {"query"}),
        }
        if not isinstance(action, str) or action not in routes:
            raise ValueError("terminal action 必须为 list_files/read_file/search")
        function, allowed, required = routes[action]
        arguments = {key: value for key, value in parameters.items() if key != "action"}
        if set(arguments) - allowed or required - set(arguments):
            raise ValueError("terminal 参数字段不符合所选 action")
        data = function(**arguments)
        return ToolResult(success=True, data=data, metadata={"tool": self.name, "action": action})

    def list_files(self, relative_path: str = ".", pattern: str = "*") -> list[str]:
        """列出匹配的普通文件，按工作区相对路径排序并截到字符预算内。

        `*` 匹配一层，`**` 匹配任意层级。逐段检查路径，禁止进入工作区外的链接。
        字符预算按返回路径的字符数之和计算，保留排序后连续的完整路径。
        """
        if not isinstance(pattern, str) or not pattern.strip() or "\x00" in pattern:
            raise ValueError("pattern 必须是有效的非空匹配模式")
        pattern_path = Path(pattern)
        if pattern_path.is_absolute() or ".." in pattern_path.parts:
            raise WorkspaceViolation("pattern 必须是工作区内的相对匹配模式")

        directory = self._resolve_path(relative_path)
        if not S_ISDIR(directory.stat().st_mode):
            raise NotADirectoryError("relative_path 必须指向目录")

        parts = pattern_path.parts
        if not parts:
            return []
        if parts[-1] == "**":
            parts += ("*",)

        matches = {directory}
        directories_only = pattern.endswith("/")
        for index, part in enumerate(parts):
            want_directory = directories_only or index < len(parts) - 1
            next_matches: set[Path] = set()
            for parent in matches:
                # **/ 只展开真实目录。其余模式逐层匹配，验证后才会进入下一层。
                for candidate in parent.glob("**/" if part == "**" else part):
                    relative = candidate.relative_to(self.workspace).as_posix()
                    resolved = self._resolve_path(relative)
                    matches_type = resolved.is_dir() if want_directory else resolved.is_file()
                    if matches_type:
                        next_matches.add(candidate)
            matches = next_matches

        if directories_only:
            return []
        paths = sorted(path.relative_to(self.workspace).as_posix() for path in matches)
        results = []
        remaining = self.max_output_chars
        for path in paths:
            if len(path) > remaining:
                break
            results.append(path)
            remaining -= len(path)
        return results

    def read_file(
        self, relative_path: str, start_line: int = 1, end_line: int | None = None
    ) -> str:
        """读取工作区内 UTF-8 文件的一基闭区间，保留原始换行和预算内的正文。"""
        resolved = self._readable_text_path(relative_path, start_line, end_line)
        lines: list[str] = []
        remaining = self.max_output_chars
        with resolved.open("r", encoding="utf-8", newline="") as file:
            for _ in range(start_line - 1):
                if not file.readline():
                    return ""

            line_number = start_line
            while remaining > 0 and (end_line is None or line_number <= end_line):
                line = file.readline(remaining)
                if not line:
                    break
                lines.append(line)
                remaining -= len(line)
                line_number += 1
        return "".join(lines)

    def read_file_page(
        self, relative_path: str, start_line: int = 1, end_line: int | None = None,
        *, max_tokens: int, count_tokens: Callable[[str], int],
    ) -> dict[str, Any]:
        """返回完整源码行。只有预算截断请求范围时才提供续页位置。"""
        resolved = self._readable_text_path(relative_path, start_line, end_line)
        lines = []
        remaining_chars = self.max_output_chars
        remaining_tokens = max_tokens
        next_line = None
        with resolved.open(encoding="utf-8", newline="") as file:
            for number, line in enumerate(islice(file, start_line - 1, None), start_line):
                if end_line is not None and number > end_line:
                    break
                cost = count_tokens(line)
                if len(line) > remaining_chars or cost > remaining_tokens:
                    if not lines:
                        raise ValueError(f"第 {number} 行超出单页预算，无法返回完整行")
                    next_line = number
                    break
                lines.append(line)
                remaining_chars -= len(line)
                remaining_tokens -= cost
        return {"path": relative_path, "start_line": start_line,
                "end_line": start_line + len(lines) - 1 if lines else None,
                "next_start_line": next_line, "text": "".join(lines)}

    def _readable_text_path(self, relative_path: str, start_line: int, end_line: int | None) -> Path:
        if type(start_line) is not int or start_line < 1:
            raise ValueError("start_line 必须是大于等于 1 的整数")
        if end_line is not None and (type(end_line) is not int or end_line < start_line):
            raise ValueError("end_line 必须为 None 或大于等于 start_line 的整数")
        resolved = self._resolve_path(relative_path)
        mode = resolved.stat().st_mode
        if S_ISDIR(mode):
            raise IsADirectoryError("relative_path 指向目录")
        if not S_ISREG(mode):
            raise ValueError("relative_path 必须指向普通文件")
        return resolved

    def search(
        self, query: str, relative_path: str = ".", limit: int = 20
    ) -> list[dict[str, Any]]:
        """在单个 UTF-8 文件或目录树中逐行进行区分大小写的字面搜索。

        按工作区相对路径和一基行号排序。结果达到条数上限，或下一条完整
        命中的路径、行号和原文超出剩余字符预算时，返回已收集的命中。
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query 必须是非空白字符串")
        if type(limit) is not int or limit <= 0:
            raise ValueError("limit 必须为正整数")

        resolved = self._resolve_path(relative_path)
        mode = resolved.stat().st_mode
        files: list[tuple[str, Path]] = []
        if S_ISREG(mode):
            files.append((resolved.relative_to(self.workspace).as_posix(), resolved))
        elif S_ISDIR(mode):
            pending_dirs = [resolved]
            while pending_dirs:
                directory = pending_dirs.pop()

                for candidate in directory.iterdir():
                    path = candidate.relative_to(self.workspace).as_posix()
                    target = self._resolve_path(path)
                    mode = target.stat().st_mode

                    if S_ISREG(mode):
                        files.append((path, target))
                    elif S_ISDIR(mode) and not candidate.is_symlink():
                        pending_dirs.append(target)
        else:
            raise ValueError("relative_path 必须指向普通文件或目录")

        results: list[dict[str, Any]] = []
        remaining = self.max_output_chars
        for path, target in sorted(files, key=lambda item: item[0]):
            with target.open("r", encoding="utf-8") as file:
                for line_number, line in enumerate(file, start=1):
                    if query not in line:
                        continue
                    cost = len(path) + len(str(line_number)) + len(line)
                    if cost > remaining:
                        return results
                    results.append({"path": path, "line": line_number, "text": line})
                    remaining -= cost
                    if len(results) == limit or remaining == 0:
                        return results

        return results

    def _resolve_path(self, relative_path: str) -> Path:
        """解析工作区相对路径，拒绝绝对路径及解析后位于工作区外的目标。"""
        if not isinstance(relative_path, str) or not relative_path.strip() or "\x00" in relative_path:
            raise ValueError("relative_path 必须是有效的非空路径字符串")
        path = Path(relative_path)
        if path.is_absolute():
            raise WorkspaceViolation("只允许工作区相对路径")
        resolved = (self.workspace / path).resolve()
        if not resolved.is_relative_to(self.workspace):
            raise WorkspaceViolation("路径指向工作区外")
        return resolved
