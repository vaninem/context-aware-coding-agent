"""校验项目目标和文件清单，并交付经过验证的文件。"""

from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

from .sandbox import DockerSandbox, SNAPSHOT_EXCLUDES, copy_source, tree_digest

MAX_PROJECT_FILES = 64
MAX_PROJECT_BYTES = 192_000


def project_path(root: Path, name: str) -> Path:
    """只接受工作区内的普通相对路径，拒绝符号链接和保留目录。"""
    if not isinstance(name, str) or not name.strip() or any(ord(char) < 32 or char == "\\" for char in name):
        raise ValueError("项目或文件路径必须是工作区内的相对路径")
    relative = PurePosixPath(name)
    if relative.is_absolute() or any(
        part == ".." or part in SNAPSHOT_EXCLUDES or part in {".agent-tests", ".env"}
        or (part.startswith(".env.") and part != ".env.example") or part.startswith(".venv-")
        for part in relative.parts
    ):
        raise ValueError("项目或文件路径超出允许范围")
    path = root / relative
    if path.resolve() != path or path.is_symlink():
        raise ValueError("项目或文件路径包含符号链接")
    return path


def update_files(current: dict[str, str], written: dict[str, str], removed: list[str], root: Path) -> dict[str, str]:
    """合并一次结构化写入，检查别名、目录冲突和大小。"""
    if not isinstance(written, dict) or not isinstance(removed, list):
        raise ValueError("files 应为路径到正文的对象，remove 应为路径数组")
    if len(written) + len(removed) > MAX_PROJECT_FILES:
        raise ValueError("一次提交的文件数超出上限")

    def file_name(name: str) -> str:
        relative = project_path(root, name).relative_to(root).as_posix()
        if relative == "." or relative != name:
            raise ValueError("文件名需要使用规范相对路径，例如 src/app.py")
        return relative

    files = current.copy()
    for name in removed:
        files.pop(file_name(name), None)
    for name, content in written.items():
        if not isinstance(content, str) or "\x00" in content:
            raise ValueError("项目文件只支持 UTF-8 文本")
        files[file_name(name)] = content
    if len(files) > MAX_PROJECT_FILES or sum(
        len(name.encode("utf-8")) + len(content.encode("utf-8")) for name, content in files.items()
    ) > MAX_PROJECT_BYTES:
        raise ValueError("项目文件数或总大小超出本轮上限")
    names = {name.casefold() for name in files}
    if len(names) != len(files) or any(
        parent.as_posix().casefold() in names for name in files for parent in PurePosixPath(name).parents
    ):
        raise ValueError("同一路径不能同时是文件和目录")
    return files


def write_files(root: Path, files: dict[str, str]) -> None:
    for name, content in files.items():
        path = project_path(root, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode("utf-8"))


def normalize_project_root(files: dict[str, str], target: str) -> dict[str, str]:
    """识别模型把整个项目又包进同名目录的输出。"""
    if target == ".":
        return files
    for prefix in dict.fromkeys((target + "/", PurePosixPath(target).name + "/")):
        if files and all(name.startswith(prefix) for name in files) and any(
            name == prefix + marker for name in files
            for marker in ("README.md", "pyproject.toml", "package.json", "Cargo.toml", "go.mod")
        ):
            return {name[len(prefix):]: body for name, body in files.items()}
    return files


def publish_project(workspace: Path, relative: str, files: dict[str, str]) -> Path:
    """原子交付新目录，恢复时识别已交付的相同文件。"""
    target = project_path(workspace, relative)
    anchor = target
    while not anchor.parent.is_dir():
        anchor = anchor.parent
    with TemporaryDirectory(prefix=".agent-build-", dir=anchor.parent) as directory:
        staged = Path(directory).resolve() / "project"
        candidate = staged / target.relative_to(anchor)
        candidate.mkdir(parents=True)
        write_files(candidate, files)
        # 再查目标，避免在运行测试期间创建的目录被覆盖。
        if project_path(workspace, relative) != target:
            raise ValueError("交付路径在验证期间发生变化")
        if anchor.exists():
            if not anchor.is_dir():
                raise ValueError(f"交付目标已被文件占用：{anchor}")
            if any(anchor.iterdir()):
                if target.is_dir() and tree_digest(target) == tree_digest(candidate):
                    return target
                raise ValueError(f"交付目标已有内容：{anchor}")
            staged.chmod(anchor.stat().st_mode & 0o777)
        staged.rename(anchor)
    return target


def apply_verified_patch(
    workspace: Path, snapshot: Path, state_dir: Path, patch: str, base_digest: str,
) -> None:
    """核对测试用的源码版本后应用补丁，支持交付已发生时恢复。"""
    if tree_digest(snapshot) != base_digest:
        raise ValueError("已验证的源码快照已发生变化")
    with TemporaryDirectory(prefix="agent-delivery-") as directory:
        output = Path(directory)
        current = output / "current"
        copy_source(workspace, current, state_dir)
        current_digest = tree_digest(current)
        if current_digest != base_digest:
            expected = output / "expected"
            copy_source(snapshot, expected, state_dir)
            DockerSandbox._git_apply(expected, patch)
            if tree_digest(expected) == current_digest:
                return
            raise ValueError("测试期间工作区已变化，已保留补丁")
        DockerSandbox._git_apply(workspace, patch)
