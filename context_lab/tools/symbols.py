"""基于已有 Python 代码块元数据查定义、直接引用和关联测试。"""

from pathlib import Path

from ..rag.expansion import build_symbol_index
from .rag import RAGTool
from .sandbox import SNAPSHOT_EXCLUDES


class PythonSymbols:
    def __init__(self, workspace: Path, state_dir: Path, *, rag: RAGTool | None = None, pattern: str = "**/*.py"):
        workspace, state_dir = workspace.resolve(), state_dir.resolve()
        rag = rag if rag is not None else RAGTool(workspace)
        # 仅排除工作区内的状态子目录。修复快照本身放在状态目录之下。
        exclude_state = state_dir != workspace and state_dir.is_relative_to(workspace)
        for path in sorted(workspace.glob(pattern)):
            relative = path.relative_to(workspace)
            if (path.suffix != ".py" or path.is_symlink() or not path.is_file()
                    or not path.resolve().is_relative_to(workspace)
                    or (exclude_state and path.is_relative_to(state_dir))
                    or any(part in SNAPSHOT_EXCLUDES or part.startswith(".venv-") for part in relative.parts)):
                continue
            try:
                rag.add_document(relative)
            except (UnicodeDecodeError, SyntaxError):
                continue
        self.rag = rag
        self.definitions = build_symbol_index(rag.chunks)

    def find(self, symbol: str, limit: int = 5) -> dict:
        """优先区分限定名，短名同时返回同名定义，引用为静态候选。"""
        symbol = symbol.strip()
        if not symbol or not all(part.isidentifier() for part in symbol.split(".")):
            raise ValueError("symbol 必须是 Python 标识符或点分限定名")
        matches = self.definitions.get(symbol, [])
        if not matches and "." in symbol:
            matches = [chunk for chunk in self.definitions.get(symbol.rsplit(".", 1)[-1], [])
                       if chunk.metadata.get("qualified_symbol", "").endswith(f".{symbol}")]
        names = {symbol, *(chunk.metadata["qualified_symbol"] for chunk in matches if "qualified_symbol" in chunk.metadata)}
        import_names = {name.removeprefix("src.") for name in names}
        result = {"definitions": [], "references": [], "tests": []}
        seen: set[tuple[str, str, int]] = set()
        definition_ids = {chunk.id for chunk in matches}
        for chunk in self.rag.chunks:
            if chunk.id in definition_ids:
                kind = "definitions"
            elif any(reference in names or reference in import_names
                     or ("." not in symbol and reference.rsplit(".", 1)[-1] == symbol)
                     or self._local_reference(reference, chunk, matches)
                     for reference in chunk.metadata.get("references", ())):
                path = Path(chunk.source)
                is_test = any(part in {"test", "tests"} for part in path.parts[:-1]) or (
                    path.name.startswith("test_") or path.name.endswith("_test.py")
                )
                kind = "tests" if is_test else "references"
            else:
                continue
            line = chunk.metadata["start_line"]
            key = (kind, chunk.source, line)
            if key not in seen and len(result[kind]) < limit:
                seen.add(key)
                result[kind].append({"path": chunk.source, "line": line,
                                     "symbol": chunk.metadata.get("qualified_symbol")})
        return result

    @staticmethod
    def _local_reference(reference, caller, definitions) -> bool:
        for target in definitions:
            if caller.source != target.source:
                continue
            name = target.metadata["symbol"]
            class_name = target.metadata.get("class_name")
            if not class_name and reference == name:
                return True
            if class_name and caller.metadata.get("class_name") == class_name and reference in (f"self.{name}", f"cls.{name}"):
                return True
        return False
