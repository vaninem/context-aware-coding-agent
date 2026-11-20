"""在临时目录用离线模型演示一次阶段摘要。"""

from pathlib import Path
from tempfile import TemporaryDirectory

from context_lab.agents import ContextAwareAgent
from context_lab.config import AppConfig, ContextConfig
from context_lab.memory import MemoryService
from context_lab.models import ScriptedChatModel
from context_lab.tools.memory import MemoryTool
from context_lab.types import Message


def main():
    with TemporaryDirectory(prefix="context-compaction-") as directory:
        config = AppConfig(state_dir=Path(directory), context=ContextConfig(auto_summary_enabled=True))
        service = MemoryService(
            config.memory_path, config.user_id,
            summary_model=ScriptedChatModel(["已定位缓存连接池，修复尚待回归验证。"]),
        )
        memory = MemoryTool(config.memory_path, config.user_id, service=service)
        agent = ContextAwareAgent(config, ScriptedChatModel(["接下来验证连接池修复。"]), memory_tool=memory)
        agent.history = [
            Message("user", "旧排查记录。" * 190), Message("assistant", "等待回归验证。" * 160),
            Message("user", "已查看连接池。"), Message("assistant", "发现长时间等待。"),
            Message("user", "先保留诊断日志。"), Message("assistant", "日志已保留。"),
        ]
        reply = agent.run("接下来检查什么？", task_id="cache-timeout")
        print(f"输入预算 {config.context.input_budget}，触发阈值 {config.context.summary_trigger_tokens}")
        print(f"阶段摘要覆盖 {agent.compaction_state.covered_until} 条历史，保留 {len(agent.history)} 条原始消息")
        print(f"本轮输入 {reply.context_result.token_count}，离线回复 {reply.answer}")


if __name__ == "__main__":
    main()
