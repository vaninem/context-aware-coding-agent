"""将已验证的补丁交付到一个新 Git 工作树。"""

import subprocess
from contextlib import suppress
from pathlib import Path
from tempfile import TemporaryDirectory

from ..types import RepairResult
from .sandbox import DockerSandbox, copy_source, tree_digest


def _git(workspace: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(workspace), *args], text=True, capture_output=True, timeout=30)
    if result.returncode != 0:
        raise ValueError(result.stderr.strip() or "Git 操作失败")
    return result.stdout.strip()


def base_commit(workspace: Path) -> str | None:
    try:
        if Path(_git(workspace, "rev-parse", "--show-toplevel")).resolve() == workspace:
            return _git(workspace, "rev-parse", "HEAD")
    except ValueError:
        pass
    return None


def deliver(workspace: Path, result: RepairResult, branch: str, destination: Path) -> Path:
    """对比已测试的源快照和 Git 基线，创建新分支工作树并暂存补丁。"""
    commit = result.get("base_commit")
    if result["status"] != "passed" or not commit or not result.get("base_digest"):
        raise ValueError("交付需要已通过验证且有 Git 提交基线的补丁")
    destination = destination.resolve()
    if destination.exists() or destination.is_relative_to(workspace) or not destination.parent.is_dir():
        raise ValueError("请指定仓库外的新路径，且其上级目录必须存在")
    if not branch or branch.startswith("-") or _git(workspace, "check-ref-format", "--branch", branch) != branch:
        raise ValueError("需要有效的新分支名")

    _git(workspace, "-c", "core.hooksPath=/dev/null", "worktree", "add", "-b", branch, str(destination), commit)
    try:
        with TemporaryDirectory(prefix="context-lab-deliver-") as folder:
            staged = Path(folder) / "baseline"
            copy_source(destination, staged, Path(folder))
            if tree_digest(staged) != result["base_digest"]:
                raise ValueError("Git 提交内容与通过测试的快照不同，请先提交要纳入测试的改动，再重新验证")
        DockerSandbox._git_apply(destination, result["patch"], "--index")
    except BaseException:
        with suppress(ValueError, OSError, subprocess.TimeoutExpired):
            _git(workspace, "worktree", "remove", "--force", str(destination))
            _git(workspace, "branch", "-D", "--", branch)
        raise
    return destination
