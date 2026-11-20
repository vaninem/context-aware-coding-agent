"""统计按检索顺序返回的源码在 token 预算内覆盖了哪些目标行。"""

import re
from bisect import bisect_left, bisect_right
from collections.abc import Iterator

from ..types import SearchHit


def target_lines(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """保留与目标字符区间相交的每一行，只要求覆盖目标区间内的字符。"""
    breaks = re.finditer(r"\r\n|[\r\n]", text[start:end])
    boundaries = [start, *(start + match.end() for match in breaks if start + match.end() < end), end]
    return list(zip(boundaries, boundaries[1:]))


def _uncovered(start: int, end: int, spans: list[tuple[int, int]]) -> Iterator[tuple[int, int]]:
    for left, right in spans:
        if right <= start:
            continue
        if left >= end:
            break
        if start < left:
            yield start, left
        start = max(start, right)
    if start < end:
        yield start, end


def _record(spans: list[tuple[int, int]], start: int, end: int) -> None:
    first = bisect_left(spans, start, key=lambda span: span[1])
    last = bisect_right(spans, end, key=lambda span: span[0])
    if first < last:
        start = min(start, spans[first][0])
        end = max(end, spans[last - 1][1])
    spans[first:last] = [(start, end)]


def covered_lines(hits: list[SearchHit], sources: dict[str, str], target: str, lines: list[tuple[int, int]],
                  budget: int, encoding) -> int:
    """去除重复的原文字符，按排名累计 o200k 源码 token，再检查完整目标行。"""
    seen: dict[str, list[tuple[int, int]]] = {}
    for hit in hits:
        chunk = hit.chunk
        text = sources[chunk.source]
        spans = seen.setdefault(chunk.source, [])
        for start, end in list(_uncovered(chunk.start, chunk.end, spans)):
            tokens = encoding.encode_ordinary(text[start:end])
            if len(tokens) > budget:
                prefix = encoding.decode_bytes(tokens[:budget]).decode("utf-8", errors="ignore")
                end = start + len(prefix)
            if start < end:
                _record(spans, start, end)
            budget -= len(tokens)
            if budget <= 0:
                break
        if budget <= 0:
            break
    target_spans = seen.get(target, [])
    return sum(any(left <= start and end <= right for left, right in target_spans) for start, end in lines)
