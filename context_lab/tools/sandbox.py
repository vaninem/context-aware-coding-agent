"""在一次性 Docker 容器里验证候选补丁。"""

import fnmatch
import hashlib
import os
import re
import selectors
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from uuid import uuid4

from ..types import RepairResult, TestRun


MAX_PATCH_CHARS = 48_000
MAX_LOG_BYTES = 8_192
DEFAULT_PROJECT_TEST_COMMAND = ("python", "-m", "unittest", "discover", "-s", "tests")
SNAPSHOT_EXCLUDES = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "node_modules", ".DS_Store"}
PROTECTED_FILES = ("test_*.py", "*_test.py", "conftest.py", "pytest.ini", ".pytest.ini", "pytest.toml", ".pytest.toml", "tox.ini", "setup.cfg",
                   "pyproject.toml", "setup.py", "Dockerfile*", "Makefile", "*.lock", "requirements*.txt",
                   "sitecustomize.py", "usercustomize.py", "unittest.py", "pytest.py")
NEW_TEST_PATH = re.compile(r"tests?/test_[A-Za-z0-9_]+\.py\Z")


def test_runner(command: tuple[str, ...]) -> str:
    """识别可由程序安全补充文件参数的 Python 测试入口。"""
    if command and Path(command[0]).name == "pytest":
        return "pytest"
    if len(command) > 2 and command[1] == "-m" and command[2] in {"pytest", "unittest"}:
        return command[2]
    return ""


def targeted_test_command(command: tuple[str, ...], path: Path) -> tuple[str, ...]:
    runner = test_runner(command)
    if runner == "pytest":
        entrypoint = command[:3] if command[1:3] == ("-m", "pytest") else command[:1]
        return (*entrypoint, "-q", f"./{path.as_posix()}")
    if runner == "unittest" and path.name.startswith("test"):
        return (*command[:3], "discover", "-s", f"./{path.parent.as_posix()}", "-p", path.name)
    raise ValueError("当前测试入口不支持指定该测试文件，可运行整套测试")


class GitApplyError(ValueError):
    """Git 拒绝解析或应用补丁。"""


def copy_source(workspace: Path, destination: Path, state_dir: Path) -> None:
    """固定本轮的输入源码，排除运行状态、环境配置和链接。"""
    def ignored(directory: str, names: list[str]) -> set[str]:
        excluded = set()
        for name in names:
            path = Path(directory) / name
            if (name in SNAPSHOT_EXCLUDES or name.startswith((".env", ".venv-")) or path == state_dir
                    or path.is_symlink() or not (path.is_file() or path.is_dir())):
                excluded.add(name)
        return excluded

    shutil.copytree(workspace, destination, ignore=ignored, symlinks=True)


def tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        digest.update(os.fsencode(path.relative_to(root).as_posix()) + b"\0")
        digest.update(b"x" if path.stat().st_mode & 0o111 else b"-")
        content = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(65536), b""):
                content.update(block)
        digest.update(content.digest())
    return digest.hexdigest()


class DockerSandbox:
    """命令和镜像由调用方配置，模型只提交统一 diff。"""

    def __init__(self, image: str, test_command: tuple[str, ...], timeout_seconds: int = 90, *,
                 regression_command: tuple[str, ...] | None = None,
                 allowed_paths: tuple[str, ...] = (), protected_tests: Path | None = None,
                 expected_failure: str | None = None, scratch_dir: Path | None = None):
        if not image or image.startswith("-") or not test_command or not test_command[0]:
            raise ValueError("需要指定镜像和测试命令")
        if type(timeout_seconds) is not int or timeout_seconds <= 0:
            raise ValueError("测试超时必须为正整数")
        self.image = image
        self.test_command = tuple(test_command)
        self.timeout_seconds = timeout_seconds
        self.regression_command = tuple(regression_command) if regression_command else None
        self.allowed_paths = tuple(allowed_paths)
        self.expected_failure = expected_failure
        self.scratch_dir = scratch_dir.resolve() if scratch_dir is not None else None
        self.protected_tests = protected_tests.resolve(strict=True) if protected_tests else None
        if self.protected_tests is not None and not self.protected_tests.is_dir():
            raise ValueError("独立测试路径必须指向目录")

    def settings(self) -> dict:
        return {"image": self.image, "command": list(self.test_command), "timeout": self.timeout_seconds,
                "regression": list(self.regression_command or ()), "allowed_paths": list(self.allowed_paths),
                "expected_failure": self.expected_failure,
                "protected_tests": str(self.protected_tests) if self.protected_tests else None}

    def ensure_ready(self) -> None:
        """确认本地 Docker 已启动，而且指定镜像已经准备好。"""
        if not shutil.which("git") or not shutil.which("docker"):
            raise RuntimeError("沙箱测试需要 Git 和 Docker CLI")
        code, output = self._run(["docker", "image", "inspect", "--format", "{{.Id}}", self.image], 10)
        if code != 0:
            raise RuntimeError(f"Docker 镜像不可用，请确认 Docker 已启动且镜像已下载或构建。\n{output}")

    def scratch(self, prefix: str) -> TemporaryDirectory:
        if self.scratch_dir is not None:
            self.scratch_dir.mkdir(parents=True, exist_ok=True)
        return TemporaryDirectory(prefix=prefix, dir=self.scratch_dir)

    @staticmethod
    def test_snapshot(source_snapshot: Path) -> Path:
        return source_snapshot.with_name(f"{source_snapshot.name}-tests")

    def freeze_tests(self, source_snapshot: Path, workspace: Path, state_dir: Path) -> None:
        if self.protected_tests is None:
            return
        if self.protected_tests.is_relative_to(workspace) or state_dir.is_relative_to(self.protected_tests):
            raise ValueError("独立测试目录需要放在目标源码和状态目录之外")
        destination = self.test_snapshot(source_snapshot)
        copy_source(self.protected_tests, destination, state_dir)
        self._prepare_container_input(destination)

    def baseline(self, snapshot: Path, *, check_passing_regression: bool = False,
                 command: tuple[str, ...] | None = None) -> dict[str, TestRun]:
        """用未修改的同一份源码分别复现问题并检查回归基线。"""
        with self.scratch("coding-agent-baseline-") as directory:
            candidate = Path(directory) / "candidate"
            shutil.copytree(snapshot, candidate, symlinks=True)
            self._prepare_container_input(candidate)
            tests = self.test_snapshot(snapshot) if self.protected_tests else None
            results = {"reproduction": self._run_test(candidate, command or self.test_command, tests)}
            if results["reproduction"]["status"] in (
                ("failed", "passed") if check_passing_regression else ("failed",)
            ) and self.regression_command:
                results["regression"] = self._run_test(candidate, self.regression_command, tests)
            return results

    def verify(self, snapshot: Path, patch: str, *, command: tuple[str, ...] | None = None,
               allow_new_tests: bool = False, require_new_tests: bool = False) -> RepairResult:
        """在临时副本应用补丁，再运行配置好的测试，保留有限的日志尾部。"""
        result: RepairResult = {"status": "patch_rejected", "patch": patch, "exit_code": None, "output": ""}
        if not patch.strip() or len(patch) > MAX_PATCH_CHARS or "\x00" in patch:
            return {**result, "patch": "", "output": "补丁为空、超长或包含非法字符"}
        if not patch.endswith("\n"):
            patch += "\n"
            result["patch"] = patch

        command = command or self.test_command
        if allow_new_tests and not test_runner(command):
            raise ValueError("当前测试入口无法独立验证新增测试文件")
        if require_new_tests and not allow_new_tests:
            return {**result, "status": "sandbox_error", "error_kind": "unsupported_test_runner",
                    "output": "新用例需要用 unittest 或 pytest 独立验证，请指定其中一种测试入口。"}
        with self.scratch("coding-agent-") as directory:
            temporary = Path(directory)
            candidate = temporary / "candidate"
            shutil.copytree(snapshot, candidate, symlinks=True)
            try:
                try:
                    changes = self._review_patch(candidate, patch, allow_new_tests=allow_new_tests)
                    self._apply_patch(candidate, patch)
                except GitApplyError as exc:
                    if not any(line.startswith(("error: corrupt patch", "error: patch fragment without header"))
                               for line in str(exc).splitlines()):
                        raise
                    changes = self._review_patch(candidate, patch, "--recount", allow_new_tests=allow_new_tests)
                    self._git_apply(candidate, patch, "--recount")
                    canonical = self._diff_files(snapshot, candidate, sorted({change["path"] for change in changes}))
                    if not canonical.strip() or len(canonical) > MAX_PATCH_CHARS:
                        raise ValueError("重算后的补丁为空或超长")
                    self._git_apply(snapshot, canonical, "--check")
                    result.update(patch=canonical, patch_recounted=True)
                result["changed_files"] = changes
            except ValueError as exc:
                return {**result, "output": str(exc), "error_kind": "git_apply" if isinstance(exc, GitApplyError) else "invalid_change"}
            except (OSError, subprocess.TimeoutExpired) as exc:
                return {**result, "status": "sandbox_error", "output": f"Git 执行失败（{type(exc).__name__}）"}

            new_tests = [Path(change["path"]) for change in changes if NEW_TEST_PATH.fullmatch(change["path"])]
            if require_new_tests and not new_tests:
                return {**result, "error_kind": "missing_test_addition",
                        "output": "本轮需要新增用例。请用单独的 tests/test_*.py 文件补齐测试，再提交完整补丁。"}
            additions = {path: (candidate / path).read_bytes() for path in new_tests}
            for path in new_tests:
                (candidate / path).unlink()
            self._prepare_container_input(candidate)
            phase = "reproduction"
            tests = self.test_snapshot(snapshot) if self.protected_tests else None
            run = self._run_test(candidate, command, tests)
            verification = {phase: run}
            if run["status"] == "passed" and self.regression_command:
                phase = "regression"
                run = self._run_test(candidate, self.regression_command, tests)
                verification[phase] = run
            if run["status"] == "passed" and additions:
                original = temporary / "original"
                shutil.copytree(snapshot, original, symlinks=True)
                for path, content in additions.items():
                    for root in (original, candidate):
                        target = root / path
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(content)
                self._prepare_container_input(original)
                self._prepare_container_input(candidate)
                for path in additions:
                    focused = targeted_test_command(command, path)
                    verification[f"new_test_before:{path}"] = self._run_test(original, focused, tests)
                    phase = f"new_test_after:{path}"
                    run = verification[phase] = self._run_test(candidate, focused, tests)
                    if run["status"] != "passed":
                        break
        status = run["status"]
        return {**result, **run, "status": "regression_failed" if phase == "regression" and status == "failed" else status,
                "phase": phase, "verification": verification}

    def verify_project(self, candidate: Path) -> TestRun:
        """验证从零生成的项目，候选目录作为只读容器输入。"""
        self.ensure_ready()
        self._prepare_container_input(candidate)
        run = self._run_test(candidate, self.test_command, None)
        if run.get("error_kind") == "no_tests":
            return {**run, "status": "failed"}
        return run

    def run_tests(self, snapshot: Path, command: tuple[str, ...], *, patch: str = "", file: str | None = None) -> TestRun:
        """在隔离副本上诊断原始或最近的候选版本。"""
        with self.scratch("coding-agent-test-") as directory:
            candidate = (Path(directory) / "candidate").resolve()
            shutil.copytree(snapshot, candidate, symlinks=True)
            if patch:
                self._apply_patch(candidate, patch)
            if file is not None:
                relative = Path(file)
                target = candidate / relative
                if (relative.is_absolute() or ".." in relative.parts or target.resolve() != target
                        or target.suffix != ".py" or not target.is_file()
                        or not (target.name.startswith("test") or target.name.endswith("_test.py"))):
                    raise ValueError("测试文件必须是快照内已有的 Python 测试文件")
                command = targeted_test_command(command, relative)
            self._prepare_container_input(candidate)
            tests = self.test_snapshot(snapshot) if self.protected_tests else None
            return self._run_test(candidate, command, tests)

    def _run_test(self, candidate: Path, command: tuple[str, ...], tests: Path | None) -> TestRun:
        name = f"context-lab-{uuid4().hex}"
        start = monotonic()
        completed = False
        try:
            code, output = self._run(self._docker_command(candidate, name, command, tests), self.timeout_seconds)
            completed = code is not None and code != 125
        except OSError as exc:
            return {"status": "sandbox_error", "exit_code": None, "output": f"Docker 启动失败（{type(exc).__name__}）"}
        finally:
            if not completed:
                self._remove(name)

        result: TestRun = {"status": "failed", "exit_code": code, "output": output,
                           "elapsed_seconds": round(monotonic() - start, 3)}
        if code is None:
            return {**result, "status": "timeout", "output": output or "沙箱测试已超时"}
        runner = test_runner(command)
        if (runner == "unittest" and any(line.startswith("Ran 0 tests in ") for line in output.splitlines())
                or runner == "pytest" and code == 5):
            return {**result, "status": "sandbox_error", "error_kind": "no_tests", "output": f"{runner} 未发现任何测试"}
        if runner == "pytest" and (re.search(r"(?:^|\n)[^\n]*: No module named pytest\s*$", output)
                                   or code == 127 and "pytest: not found" in output):
            return {**result, "status": "sandbox_error", "error_kind": "missing_dependency",
                    "output": f"沙箱环境缺少模块。请核对项目依赖是否已安装到 Docker 镜像。\n{output}"}
        if code in (125, 126, 127) or code < 0 or code >= 128:
            status = "sandbox_error"
        else:
            status = "passed" if code == 0 else "failed"
        return {**result, "status": status}

    def patch_from_edits(self, snapshot: Path, edits: list[dict], *, allow_create: bool = False,
                         allow_new_tests: bool = False) -> str:
        """在同一个原始快照上定位不重叠的精确替换，并由 Git 生成标准补丁。"""
        if not isinstance(edits, list) or not edits:
            raise ValueError("edits 必须是非空列表")
        snapshot = snapshot.resolve()
        originals: dict[str, str] = {}
        additions: dict[str, str] = {}
        replacements: dict[str, list[tuple[int, int, str]]] = {}
        total_chars = 0
        for edit in edits:
            if (not isinstance(edit, dict) or set(edit) != {"path", "old_text", "new_text"}
                    or any(not isinstance(value, str) or "\x00" in value for value in edit.values())):
                raise ValueError("每项修改需要 path、old_text、new_text 三个字符串")
            old, new = edit["old_text"], edit["new_text"]
            total_chars += sum(len(value) for value in edit.values())
            if total_chars > MAX_PATCH_CHARS:
                raise ValueError("修改文本超出提交预算")
            relative = self._check_change_path(edit["path"], snapshot if allow_new_tests and old == "" else None)
            name = relative.as_posix()
            source = snapshot / relative
            if allow_create and old == "" and not source.exists() and source.resolve() == source:
                if name in additions or not new:
                    raise ValueError(f"新文件需要非空正文，且只能提交一次：{name}")
                additions[name] = new
                continue
            if not old.strip():
                raise ValueError("old_text 需要包含可定位的非空源码")
            if old == new:
                continue
            if source.resolve() != source or not source.is_file():
                raise ValueError(f"编辑目标必须是快照中的普通文件：{name}")
            if name not in originals:
                originals[name] = source.read_bytes().decode("utf-8")
            text = originals[name]
            start = text.find(old)
            if start < 0 or text.find(old, start + 1) >= 0:
                reason = "找不到原文" if start < 0 else "原文匹配多处"
                raise ValueError(f"{name}：{reason}，请读取原始快照并提交精确的唯一上下文")
            replacements.setdefault(name, []).append((start, start + len(old), new))

        if not (replacements or additions):
            raise ValueError("所有 old_text 和 new_text 都相同，源码无变化，请提交实际修改")

        with TemporaryDirectory(prefix="coding-agent-edits-") as directory:
            candidate = Path(directory).resolve()
            for name, content in additions.items():
                target = candidate / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content.encode("utf-8"))
            for name, intervals in replacements.items():
                text, previous_start = originals[name], len(originals[name])
                for start, end, new in sorted(intervals, reverse=True):
                    if end > previous_start:
                        raise ValueError(f"{name} 的修改区间重叠")
                    text = text[:start] + new + text[end:]
                    previous_start = start
                target = candidate / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(text.encode("utf-8"))
                target.chmod((snapshot / name).stat().st_mode & 0o777)
            return self._diff_files(snapshot, candidate, sorted(replacements.keys() | additions.keys()))

    @staticmethod
    def _diff_files(snapshot: Path, candidate: Path, paths: list[str]) -> str:
        """让 Git 计算区段和文件模式，将临时文件名换成仓库内路径。"""
        def quote(path: str) -> bytes:
            escaped = b"".join(bytes([byte]) if 32 <= byte < 127 and byte not in (34, 92)
                               else f"\\{byte:03o}".encode("ascii") for byte in os.fsencode(path))
            return b'"' + escaped + b'"'

        patches = []
        for name in paths:
            before, after = snapshot / name, candidate / name
            old = str(before) if before.is_file() else os.devnull
            new = str(after) if after.is_file() else os.devnull
            diff = subprocess.run(["git", "diff", "--no-index", "--no-ext-diff", "--no-textconv", "--no-color",
                                   "--no-renames", "--", old, new], cwd=candidate.parent,
                                  env=DockerSandbox._git_env(candidate), capture_output=True, timeout=10)
            if diff.returncode == 0:
                continue
            if diff.returncode != 1 or not diff.stdout.startswith(b"diff --git "):
                raise GitApplyError("Git 无法生成标准补丁")
            lines = diff.stdout.split(b"\n")
            old_path, new_path = quote(f"a/{name}"), quote(f"b/{name}")
            lines[0] = b"diff --git " + old_path + b" " + new_path
            for index, line in enumerate(lines[1:], start=1):
                if line.startswith(b"@@"):
                    break
                if line.startswith(b"--- ") and old != os.devnull:
                    lines[index] = b"--- " + old_path
                if line.startswith(b"+++ ") and new != os.devnull:
                    lines[index] = b"+++ " + new_path
            patches.append(b"\n".join(lines))
        return b"".join(patches).decode("utf-8")

    def _check_change_path(self, name: str, new_test_root: Path | None = None) -> Path:
        item = Path(name)
        if not name or not item.parts or item.is_absolute() or ".." in item.parts or "\x00" in name:
            raise ValueError("补丁包含无效路径")
        new_test = False
        if new_test_root is not None and NEW_TEST_PATH.fullmatch(name):
            path = new_test_root.resolve() / item
            new_test = not path.exists() and path.resolve() == path
        if not new_test and (any(part.startswith((".git", ".env")) or part in {"test", "tests", ".agent-tests", "unittest", "pytest"}
                for part in item.parts) or any(fnmatch.fnmatch(item.name, pattern) for pattern in PROTECTED_FILES)):
            raise ValueError(f"补丁触及测试或受保护配置：{name}")
        if self.allowed_paths and not any(fnmatch.fnmatchcase(item.as_posix(), pattern) for pattern in self.allowed_paths):
            raise ValueError(f"补丁超出允许的修改范围：{name}")
        return item

    def _review_patch(self, candidate: Path, patch: str, *options: str, allow_new_tests: bool = False) -> list[dict]:
        changes = []
        summary = self._git_apply(candidate, patch, "--summary", *options)
        operations = (row.strip().split(maxsplit=5) for row in summary.splitlines())
        unsafe_modes = {b"160000", b"120000"}
        if any(parts and (parts[0] in (b"rename", b"copy")
                          or unsafe_modes.intersection(parts[2:4] if parts[0] == b"mode" else parts[2:3]))
               for parts in operations):
            raise ValueError("补丁包含重命名、复制、子模块或符号链接变更")
        for row in self._git_apply(candidate, patch, "--numstat", "-z", *options).split(b"\x00"):
            if not row:
                continue
            added, removed, raw_path = row.split(b"\t", 2)
            path = raw_path.decode("utf-8")
            if not added.isdigit() or not removed.isdigit():
                raise ValueError("补丁包含重命名或二进制变更")
            self._check_change_path(path, candidate if allow_new_tests and int(removed) == 0 else None)
            changes.append({"path": path, "added": int(added), "removed": int(removed)})
        if not changes:
            raise ValueError("补丁没有可审查的源码改动")
        return changes

    @staticmethod
    def _apply_patch(candidate: Path, patch: str) -> None:
        DockerSandbox._git_apply(candidate, patch)

    @staticmethod
    def _git_env(candidate: Path) -> dict[str, str]:
        return {"PATH": os.environ.get("PATH", os.defpath), "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_CEILING_DIRECTORIES": str(candidate.parent), "LC_ALL": "C"}

    @staticmethod
    def _git_apply(candidate: Path, patch: str, *options: str) -> bytes:
        command = ["git", "-c", "apply.ignoreWhitespace=no", "apply", "--whitespace=nowarn", *options, "-"]
        applied = subprocess.run(command, cwd=candidate, env=DockerSandbox._git_env(candidate), input=patch.encode("utf-8"),
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10, check=False)
        if applied.returncode != 0:
            raise GitApplyError(applied.stderr[-MAX_LOG_BYTES:].decode("utf-8", errors="replace").strip())
        return applied.stdout

    @staticmethod
    def _prepare_container_input(candidate: Path) -> None:
        # 容器用户可能与宿主的文件所有者不同。只调整临时副本的读取和执行权限。
        for directory, _, filenames in os.walk(candidate):
            directory_path = Path(directory)
            directory_path.chmod(0o755)
            for name in filenames:
                path = directory_path / name
                if not path.is_symlink():
                    path.chmod(0o755 if path.stat().st_mode & 0o111 else 0o644)

    def _docker_command(self, candidate: Path, name: str, command: tuple[str, ...], tests: Path | None) -> list[str]:
        test_mount = []
        if tests is not None:
            test_mount = ["--mount", f"type=bind,source={tests},target=/agent-tests,readonly"]
        return [
            "docker", "run", "--rm", "--name", name, "--pull=never", "--network=none",
            "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges", "--user=1000:1000",
            "--memory=512m", "--memory-swap=512m", "--cpus=1", "--pids-limit=128", "--log-driver=none",
            "--mount", f"type=bind,source={candidate},target=/input,readonly",
            *test_mount,
            "--tmpfs", "/work:rw,exec,size=256m,mode=1777", "--tmpfs", "/tmp:rw,size=64m,mode=1777",
            "--workdir", "/work", "--env", "HOME=/tmp", "--env", "PYTHONDONTWRITEBYTECODE=1", "--env", "PYTHONPATH=/work",
            "--entrypoint", "/bin/sh", self.image,
            "-ec", 'cp -R /input/. . || exit 125; exec "$@"', "sandbox", *command,
        ]

    @staticmethod
    def _run(command: list[str], timeout: int) -> tuple[int | None, str]:
        """持续读取管道，避免子进程阻塞，只保留最近的输出。"""
        output = bytearray()
        deadline = monotonic() + timeout
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              stdin=subprocess.DEVNULL) as process:
            try:
                assert process.stdout is not None
                with selectors.DefaultSelector() as selector:
                    selector.register(process.stdout, selectors.EVENT_READ)
                    while True:
                        remaining = deadline - monotonic()
                        if remaining <= 0 or not selector.select(remaining):
                            raise subprocess.TimeoutExpired(command, timeout)
                        data = os.read(process.stdout.fileno(), MAX_LOG_BYTES)
                        if not data:
                            break
                        output.extend(data)
                        del output[:-MAX_LOG_BYTES]
                code = process.wait(timeout=max(0, deadline - monotonic()))
            except subprocess.TimeoutExpired:
                process.kill()
                code = None
            return code, output.decode("utf-8", errors="replace").strip()

    @staticmethod
    def _remove(name: str) -> None:
        try:
            subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=10, check=False)
        except (OSError, subprocess.TimeoutExpired):
            pass
