"""组装、计数和裁剪发给模型的上下文。"""

from datetime import datetime
from typing import Any
from ..config import ContextConfig
from ..errors import ContextBudgetError, ExercisePending
from ..memory.retrieval import LexicalMemoryRetriever
from ..ports import KnowledgeRetriever, MemoryRetriever
from ..tokens import ApproxTokenCounter
from ..types import ContextBuildResult, ContextCompactionState, ContextPacket, ContextUsage, Message, utc_now


class ContextBuilder:
    OUTPUT_INSTRUCTION = "请依据可核查证据回答；区分事实、推测与待办。"
    HEADINGS = ("[Role & Policies]", "[Task]", "[State]", "[Evidence]", "[Context]", "[Output]")

    def __init__(
        self,
        config: ContextConfig | None = None,
        memory_tool: MemoryRetriever | None = None,
        rag_tool: KnowledgeRetriever | None = None,
        token_counter: Any = None,
    ) -> None:
        self.config = config if config is not None else ContextConfig()
        self.memory_tool = memory_tool
        self.rag_tool = rag_tool
        self.token_counter = token_counter if token_counter is not None else ApproxTokenCounter()
        self.last_warnings: list[str] = []

    def inspect_usage(
        self,
        user_query: str,
        conversation_history: list[Message] | None = None,
        system_instructions: str | None = None,
        custom_packets: list[ContextPacket] | None = None,
        *,
        state: ContextCompactionState | None = None,
        summary_packet: ContextPacket | None = None,
    ) -> ContextUsage:
        """估算裁剪前的整轮输入，并在完整交互边界规划旧历史摘要。"""
        history = conversation_history if conversation_history is not None else []
        start = state.covered_until if state is not None else 0
        if type(start) is not int or not 0 <= start <= len(history):
            raise ValueError("历史摘要覆盖范围无效")
        if start and summary_packet is None:
            raise ValueError("已覆盖的历史需要对应阶段摘要")
        packets = list(custom_packets or [])
        if summary_packet is not None:
            packets.append(summary_packet)
        estimated = self.estimate_input(user_query, history[start:], system_instructions, packets)
        budget = self.config.input_budget
        trigger = self.config.summary_trigger_tokens
        if estimated < trigger:
            return ContextUsage(estimated, budget, trigger, False, start, start, "below_threshold")

        latest_cut = len(history) - self.config.summary_keep_recent_messages
        end = start
        if start < latest_cut and history[start].role == "user":
            end = next((
                index for index in range(latest_cut, start, -1)
                if history[index].role == "user" and history[index - 1].role == "assistant"
            ), start)
        if end == start:
            return ContextUsage(estimated, budget, trigger, False, start, start, "no_complete_old_turn")
        remaining = self.estimate_input(user_query, history[end:], system_instructions, custom_packets)
        if remaining >= budget:
            return ContextUsage(estimated, budget, trigger, False, start, start, "remaining_input_exhausts_budget")
        return ContextUsage(estimated, budget, trigger, True, start, end, "threshold_reached")

    def estimate_input(
        self, user_query: str, history: list[Message], system_instructions: str | None,
        packets: list[ContextPacket] | None = None,
    ) -> int:
        """使用最终的区块格式计数，保留完整的传入历史。"""
        inputs = self._gather(user_query, history, system_instructions, packets,
                              sources_prepared=True, keep_full_history=True)
        return self._count_tokens(self._structure(inputs, user_query))

    def build(
        self,
        user_query: str,
        conversation_history: list[Message] | None = None,
        system_instructions: str | None = None,
        custom_packets: list[ContextPacket] | None = None,
        now: datetime | None = None,
        *,
        sources_prepared: bool = False,
    ) -> ContextBuildResult:
        """收集、排序和渲染输入，并核对包含区块标题的最终预算。"""
        if not user_query.strip():
            raise ValueError("user_query 不能为空")
        self.last_warnings = []
        current_time = now if now is not None else utc_now()
        candidates = self._gather(user_query, conversation_history, system_instructions, custom_packets,
                                  sources_prepared=sources_prepared)
        selected = self._select(
            candidates, user_query, self.config.input_budget, now=current_time
        )
        context = self._structure(selected, user_query)
        if self._count_tokens(context) > self.config.input_budget:
            if not self.config.enable_compression:
                raise ContextBudgetError("结构化后的上下文超限，且未启用裁剪")
            while self._count_tokens(context) > self.config.input_budget:
                optional_index = next((
                    index for index in range(len(selected) - 1, -1, -1)
                    if not self._required(selected[index])
                ), None)
                if optional_index is None:
                    raise ContextBudgetError("系统指令、当前任务与必要上下文超出输入预算")
                selected.pop(optional_index)
                context = self._structure(selected, user_query)
        token_count = self._count_tokens(context)
        if token_count > self.config.input_budget:
            raise ContextBudgetError("压缩后仍超预算：应复算标题、分隔符和提示标记")
        return ContextBuildResult(
            context=context,
            candidates=candidates,
            selected=selected,
            token_count=token_count,
            warnings=list(self.last_warnings),
        )

    def _count_tokens(self, text: str) -> int:
        return self.token_counter.count(text)

    def _gather(
        self,
        user_query: str,
        conversation_history: list[Message] | None = None,
        system_instructions: str | None = None,
        custom_packets: list[ContextPacket] | None = None,
        *,
        sources_prepared: bool = False,
        keep_full_history: bool = False,
    ) -> list[ContextPacket]:
        """封装检索资料、历史及系统指令，保留各自的来源和权限。"""
        packets = []
        if system_instructions:
            packets.append(ContextPacket(system_instructions, metadata={"type": "system_instruction"}))
        history = conversation_history if conversation_history is not None else []
        if not (keep_full_history or self.config.auto_summary_enabled):
            limit = self.config.history_limit
            history = history[-limit:] if limit else []
        packets.extend(ContextPacket(message.content, timestamp=message.timestamp,
                                     metadata={"type": "conversation_history", "role": message.role})
                       for message in history)
        if sources_prepared:
            packets.extend(custom_packets or [])
        else:
            packets.extend(self.collect_sources(user_query, custom_packets, warnings=self.last_warnings))
        return packets

    def collect_sources(
        self, user_query: str, custom_packets: list[ContextPacket] | None = None,
        *, warnings: list[str] | None = None, include_memory: bool = True,
    ) -> list[ContextPacket]:
        """检索一次本轮资料，供预算检查和最终组装共用。"""
        packets = []
        for kind, retriever in (("memory", self.memory_tool if include_memory else None), ("rag_result", self.rag_tool)):
            if retriever is None:
                continue
            try:
                for hit in retriever.search(user_query, limit=5):
                    if kind == "memory":
                        packets.append(ContextPacket(hit.content, timestamp=hit.created_at, metadata={
                            "type": kind, "id": hit.id, "memory_type": hit.memory_type, "user_id": hit.user_id,
                            "valid_from": hit.valid_from, "valid_to": hit.valid_to, "importance": hit.importance,
                            "history_id": hit.metadata.get("history_id"),
                        }))
                    else:
                        chunk = hit.chunk
                        evidence = hit.context_chunk or chunk
                        packets.append(ContextPacket(evidence.content, metadata={
                            "type": kind, "id": chunk.id, "source": chunk.source,
                            "heading_path": chunk.heading_path, "start": chunk.start, "end": chunk.end,
                            "start_line": chunk.metadata.get("start_line"), "end_line": chunk.metadata.get("end_line"),
                            "context_start": evidence.start, "context_end": evidence.end,
                        }))
            except ExercisePending:
                raise
            except (OSError, RuntimeError) as exc:
                if warnings is not None:
                    warnings.append(f"{kind} 检索失败（{type(exc).__name__}）")
        packets.extend(custom_packets or [])
        return packets

    def _calculate_relevance(self, content: str, query: str) -> float:
        """复用记忆检索的中英文词项，用 Jaccard 计算词法相关性。"""
        tokenizer = LexicalMemoryRetriever()
        left, right = set(tokenizer.tokenize(content)), set(tokenizer.tokenize(query))
        union = left | right
        return len(left & right) / len(union) if union else 0.0

    def _calculate_recency(
        self, timestamp: datetime, now: datetime | None = None
    ) -> float:
        """以七天为半衰期计算近因分数，未来时间按当前处理。"""
        current = now if now is not None else utc_now()
        if timestamp.utcoffset() is None or current.utcoffset() is None:
            raise ValueError("上下文时间必须带时区")
        age = max(0.0, (current - timestamp).total_seconds())
        return 2 ** (-age / (7 * 86400))

    def _required(self, packet: ContextPacket) -> bool:
        kind = packet.metadata.get("type")
        return kind in ("system_instruction", "progress_summary") or (
            self.config.auto_summary_enabled and kind == "conversation_history"
        )

    def _select(
        self,
        packets: list[ContextPacket],
        user_query: str,
        available_tokens: int,
        now: datetime | None = None,
    ) -> list[ContextPacket]:
        """保留必需资料，对可选资料按词法相关性和近因性排序。"""
        required = []
        ranked = []
        for packet in packets:
            packet.token_count = self._count_tokens(packet.content)
            if self._required(packet):
                required.append(packet)
                continue
            relevance = packet.relevance_score
            if relevance is None:
                relevance = self._calculate_relevance(packet.content, user_query)
            if relevance < self.config.min_relevance:
                continue
            score = self.config.relevance_weight * relevance + self.config.recency_weight * (
                self._calculate_recency(packet.timestamp, now)
            )
            ranked.append((score, packet))
        remaining = available_tokens - sum(packet.token_count for packet in required)
        if remaining < 0:
            raise ContextBudgetError("必要上下文超出输入预算")
        required.sort(key=lambda packet: packet.metadata.get("type") != "system_instruction")
        for _, packet in sorted(ranked, key=lambda entry: -entry[0]):
            if packet.token_count <= remaining:
                required.append(packet)
                remaining -= packet.token_count
        return required

    def _structure(self, selected_packets: list[ContextPacket], user_query: str) -> str:
        """固定渲染六个区块，保留历史角色和资料的低权限位置。"""
        sections = {heading: [] for heading in self.HEADINGS}
        sections["[Task]"].append(self._escape_headings(user_query))
        sections["[Output]"].append(self.OUTPUT_INSTRUCTION)
        for packet in selected_packets:
            kind = packet.metadata.get("type")
            body = self._escape_headings(packet.content)
            if kind == "system_instruction":
                section = "[Role & Policies]"
            elif kind == "rag_result":
                section = "[Evidence]"
            elif kind == "note" and packet.metadata.get("note_type") in ("task_state", "action", "blocker"):
                section = "[State]"
            else:
                section = "[Context]"
                if kind == "conversation_history":
                    body = f"{packet.metadata['role']}: {body}"
                elif kind == "progress_summary":
                    body = f"阶段摘要\n{body}"
            sections[section].append(body)
        return "\n\n".join(heading + "\n" + "\n\n".join(sections[heading]) for heading in self.HEADINGS)

    def _escape_headings(self, text: str) -> str:
        return "\n".join("> " + line if line in self.HEADINGS else line for line in text.split("\n"))

    def _compress(self, context: str, max_tokens: int) -> str:
        """裁剪已渲染的可选正文，保留系统策略、任务与输出说明。"""
        if self._count_tokens(context) <= max_tokens:
            return context
        sections = {heading: [] for heading in self.HEADINGS}
        heading = None
        for line in context.split("\n"):
            if line in sections:
                heading = line
            elif heading is not None:
                sections[heading].append(line)
        required = ("[Role & Policies]", "[Task]", "[Output]")
        packed = {
            key: "\n".join(sections[key]).rstrip("\n") if key in required else ""
            for key in self.HEADINGS
        }
        for key in (None, "[State]", "[Evidence]", "[Context]"):
            base = "\n\n".join(title + "\n" + packed[title] for title in self.HEADINGS)
            remaining = max_tokens - self._count_tokens(base)
            if remaining < 0:
                raise ContextBudgetError("系统策略、当前任务与输出说明超出输入预算")
            if key is not None:
                packed[key] = self._truncate_text("\n".join(sections[key]).rstrip("\n"), remaining)
        result = "\n\n".join(title + "\n" + packed[title] for title in self.HEADINGS)
        if self._count_tokens(result) > max_tokens:
            raise ContextBudgetError("压缩后的上下文仍超出预算")
        return result

    def _truncate_text(self, text: str, max_tokens: int) -> str:
        return self.token_counter.truncate(text, max_tokens)
