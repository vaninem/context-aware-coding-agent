"""已提供的本地 JSON 基础设施；它不包含记忆检索或笔记索引的答案。"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any


class JsonFileStore:
    """读取 UTF-8 JSON，并在同一目录通过临时文件替换写入。

    只有文件不存在时才返回默认值。坏 JSON、编码错误和文件系统错误继续抛出，
    方便调用者分清“初次运行”和“已有数据损坏”。本类没有并发写锁；多个进程
    同时改写同一文件不属于本练习的保证范围。
    """

    def __init__(self, path: Path):
        """保存文件位置；初始化不会创建目录、读取文件或发起网络请求。"""
        self.path = Path(path)

    def read(self, default: Any = None) -> Any:
        """返回解析后的 JSON；仅 FileNotFoundError 表示使用 default。

        JSON 的 null、空列表或空对象都是有效数据，不会被替换成默认值。
        """
        try:
            with self.path.open("r", encoding="utf-8") as source:
                return json.load(source)
        except FileNotFoundError:
            return default

    def write(self, value: Any) -> None:
        """写入完整 JSON 后替换目标；序列化或写入失败不报告成功。

        临时文件位于目标父目录，替换不会跨文件系统。拒绝 NaN/Infinity 等
        非标准 JSON 数值；失败时清理临时文件并保留原异常。
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as destination:
                temporary_path = Path(destination.name)
                json.dump(value, destination, ensure_ascii=False, indent=2, allow_nan=False)
                destination.write("\n")
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(temporary_path, self.path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
