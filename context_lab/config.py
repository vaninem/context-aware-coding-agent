"""配置只保存参数和验证约束，不进行网络调用。"""

from dataclasses import dataclass, field
from math import ceil, isclose, isfinite
from pathlib import Path


ECNU_MAX_CONTEXT_TOKENS = 512_000
ECNU_MAX_OUTPUT_TOKENS = 256_000
ECNU_MAX_APP_INPUT_TOKENS = 32_768
ECNU_MAX_APP_OUTPUT_TOKENS = 8_192


@dataclass
class ContextConfig:
    max_tokens: int = 3000
    reserve_ratio: float = 0.2  # 预留给回答的空间，最终输入预算为 max_tokens*(1-ratio)。
    min_relevance: float = 0.1
    relevance_weight: float = 0.7
    recency_weight: float = 0.3
    history_limit: int = 5
    enable_compression: bool = True
    auto_summary_enabled: bool = False
    summary_trigger_ratio: float = 0.8
    summary_target_ratio: float = 0.6
    summary_keep_recent_messages: int = 4  # 消息条数，切点仍须保留完整交互
    summary_max_attempts_per_turn: int = 1

    def __post_init__(self) -> None:
        if self.max_tokens <= 0 or self.history_limit < 0:
            raise ValueError("max_tokens 要为正数，history_limit 不能为负")
        if not 0 <= self.reserve_ratio < 1 or not 0 <= self.min_relevance <= 1:
            raise ValueError("预留比例和相关性阈值越界")
        if min(self.relevance_weight, self.recency_weight) < 0 or not isclose(
            self.relevance_weight + self.recency_weight, 1.0
        ):
            raise ValueError("评分权重非负且和为 1")
        if type(self.auto_summary_enabled) is not bool:
            raise ValueError("auto_summary_enabled 必须是 bool")
        ratios = (self.summary_trigger_ratio, self.summary_target_ratio)
        if any(type(value) not in (int, float) or not isfinite(value) for value in ratios):
            raise ValueError("摘要阈值必须是有限数值")
        if not 0 < self.summary_target_ratio < self.summary_trigger_ratio <= 1:
            raise ValueError("摘要比例必须满足 0 < 目标 < 触发阈值 <= 1")
        for name in ("summary_keep_recent_messages", "summary_max_attempts_per_turn"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} 必须是正整数")

    @property
    def input_budget(self) -> int:
        return int(self.max_tokens * (1 - self.reserve_ratio))

    @property
    def summary_trigger_tokens(self) -> int:
        return ceil(self.input_budget * self.summary_trigger_ratio)

    @property
    def summary_target_tokens(self) -> int:
        return int(self.input_budget * self.summary_target_ratio)


@dataclass
class AppConfig:
    project_name: str = "demo-project"
    workspace: Path = Path("examples/sample_project")
    state_dir: Path = Path("runtime/demo-project")
    user_id: str = "student"
    context: ContextConfig = field(default_factory=ContextConfig)
    terminal_max_chars: int = 8000
    auto_memory_write_enabled: bool = False
    max_output_tokens: int | None = None

    @classmethod
    def for_model(cls, model_name: str, **kwargs) -> "AppConfig":
        """装配 ecnu-max 的工作预算，其他模型采用调用方指定的配置。"""
        if model_name.strip() == "ecnu-max":
            kwargs.setdefault("context", ContextConfig(max_tokens=ECNU_MAX_APP_INPUT_TOKENS + ECNU_MAX_APP_OUTPUT_TOKENS))
            budget = kwargs["context"]
            if budget.max_tokens > ECNU_MAX_CONTEXT_TOKENS:
                raise ValueError("ecnu-max 的总预算不得超过 512000 token")
            kwargs.setdefault("terminal_max_chars", 24_000)
            reserve = budget.max_tokens - budget.input_budget
            kwargs.setdefault("max_output_tokens", min(ECNU_MAX_APP_OUTPUT_TOKENS, reserve) or None)
            if (kwargs["max_output_tokens"] or 0) > ECNU_MAX_OUTPUT_TOKENS:
                raise ValueError("ecnu-max 的输出预算不得超过 256000 token")
        return cls(**kwargs)

    def __post_init__(self) -> None:
        self.workspace = Path(self.workspace).resolve()
        self.state_dir = Path(self.state_dir).resolve()
        if self.terminal_max_chars <= 0:
            raise ValueError("terminal_max_chars 要为正数")
        if self.max_output_tokens is not None and (
            type(self.max_output_tokens) is not int
            or not 0 < self.max_output_tokens <= self.context.max_tokens - self.context.input_budget
        ):
            raise ValueError("最大输出量必须放得进上下文预留预算")
        if type(self.auto_memory_write_enabled) is not bool:
            raise ValueError("auto_memory_write_enabled 必须是 bool")

    @property
    def notes_dir(self) -> Path:
        return self.state_dir / "notes"

    @property
    def memory_path(self) -> Path:
        return self.state_dir / "memories.json"

    @property
    def checkpoint_path(self) -> Path:
        return self.state_dir / "checkpoint.json"

    @property
    def workflow_checkpoint_path(self) -> Path:
        return self.state_dir / "workflow.sqlite3"
