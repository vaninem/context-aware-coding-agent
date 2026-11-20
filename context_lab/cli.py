"""从类装配开始，而不是一上来要求数据库或模型密钥。"""

import argparse
import importlib
import json
import os
import shlex
from pathlib import Path
import sys
import unittest
from hashlib import sha256
from .agents import CodebaseMaintainer
from .config import AppConfig
from .errors import ExercisePending
from .models import HTTPChatModel, ScriptedChatModel
from .storage.env_file import read_env_file
from .tools.delivery import deliver
from .tools.memory import MemoryTool
from .tools.rag import RAGTool
from .tools.sandbox import DEFAULT_PROJECT_TEST_COMMAND, DockerSandbox
from .terminal_ui import display_progress, repair_report, run_terminal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_STAGES = (
    "scaffold", "tokens", "notes", "retrieval", "terminal", "context", "agent", "repair", "embeddings",
    "memory_scaffold", "memory_codec", "memory_repository", "memory_service", "memory_policy",
    "memory_advanced", "memory_embedding", "memory_retention_scaffold", "memory_retention",
    "compaction_scaffold", "compaction", "rag_framework", "rag_exercises",
)
EXAMPLES = {
    "notes": "notes_demo", "memory-rag": "memory_rag_demo",
    "terminal": "terminal_demo", "context": "context_builder_demo",
    "maintainer": "maintainer_demo", "resume": "resume_demo", "embeddings": "embeddings_demo",
    "memory-framework": "memory_framework_demo", "memory-advanced": "memory_advanced_demo",
    "memory-cleanup-preview": "memory_cleanup_preview", "memory-cleanup-worker": "memory_cleanup_worker",
    "compaction-framework": "compaction_framework_demo",
}


def make_demo_agent(
    workspace: Path | None = None, state_dir: Path | None = None, *, sample_rag: bool = False,
    resume: bool = False,
) -> CodebaseMaintainer:
    config = AppConfig(
        workspace=workspace or PROJECT_ROOT / "examples" / "sample_project",
        state_dir=state_dir or PROJECT_ROOT / "runtime" / "demo-project",
    )
    replies = ["这是预设的离线回复，用于检查上下文和调用流程，不是真实模型的分析。"] * 5
    rag = None
    if sample_rag and workspace is None:
        rag = RAGTool(config.workspace)
        for name in ("README.md", "src/search.py", "docs/task.md"):
            rag.add_document(Path(name))
        if not resume:
            replies.insert(0, '{"action":"rag_search","arguments":{"query":"normalize_query","limit":2}}')
    return CodebaseMaintainer(config=config, model=ScriptedChatModel(replies), rag_tool=rag)


def overview(agent: CodebaseMaintainer) -> None:
    print("上下文工程练习骨架 v0.1.0")
    print(f"项目：{agent.config.project_name}")
    print(f"只读样例工作区：{agent.config.workspace}")
    print(f"运行记录目录：{agent.config.state_dir}")
    print(f"输入预算：{agent.config.context.input_budget} 个近似 token")
    print("已装配：")
    for name in ("model", "memory_tool", "note_tool", "terminal_tool", "context_builder"):
        print(f"  {name}: {type(getattr(agent, name)).__name__}")
    print("LangGraph 编排已接入。使用 repair 可以在 Docker 中验证候选补丁。")
    print("从 docs/学习路线.md 开始。overview 成功只表示初始化完成。")
    print("Memory M01—M08 已实现，详见 docs/Memory学习路线.md。")
    print("运行 python run.py example compaction-framework 可演示 80% 阈值阶段摘要。")


def configure_retrieval(config, model, mode, env_file=PROJECT_ROOT / ".env") -> dict:
    if mode == "lexical":
        return {}
    try:
        from .embeddings import EmbeddingConfig, IncrementalEmbeddingIndex
        from .memory import MemoryService
        from .memory.config import MemoryConfig
        from .memory.embedding_bridge import EmbeddingMemoryBridge
    except ImportError:
        raise RuntimeError("向量检索需要安装项目的 embedding 可选依赖") from None
    embedding = EmbeddingConfig.from_env(env_file)
    if not embedding.enabled or not embedding.database_configured:
        raise ValueError("向量检索需要配置 EMBEDDING_BASE_URL、EMBEDDING_MODEL、EMBEDDING_API_KEY 和 DATABASE_URL")
    index = IncrementalEmbeddingIndex(embedding)
    index.initialize()
    memory_config = MemoryConfig(search_mode="hybrid")
    bridge = EmbeddingMemoryBridge(index, config.user_id, memory_config, scope=str(config.memory_path))
    service = MemoryService(
        config.memory_path, config.user_id, config=memory_config, embedding=bridge, summary_model=model,
    )
    return {
        "embedding_index": index, "repair_rag_mode": mode,
        "memory_tool": MemoryTool(config.memory_path, config.user_id, service=service),
    }


def build_repair_agent(args: argparse.Namespace, *, interactive: bool = False) -> CodebaseMaintainer:
    workspace = args.workspace or PROJECT_ROOT / "examples" / "sample_project"
    user_id = args.user_id if interactive else "student"
    if interactive:
        scope = str(workspace.resolve())
        if user_id != "student":
            scope += "\n" + user_id
        identity = sha256(scope.encode()).hexdigest()[:16]
        default_state = PROJECT_ROOT / "runtime" / "terminal" / identity
    else:
        default_state = PROJECT_ROOT / "runtime" / "repair"
    state_dir = args.state_dir or default_state
    if args.sample:
        config = AppConfig(workspace=workspace, state_dir=state_dir, user_id=user_id)
        replies = []
        if args.retrieval != "lexical":
            replies.append(json.dumps({"action": "rag_search", "arguments": {"query": "normalize_query", "limit": 2}}))
        for replacement in ("return query.lstrip().lower()", "return query.strip().lower()"):
            patch = ("--- a/src/search.py\n+++ b/src/search.py\n@@ -4,3 +4,3 @@\n"
                     " def normalize_query(query: str) -> str:\n-    return query.lower()\n"
                     f"+    {replacement}\n \n")
            replies.append(json.dumps({"action": "verify_patch", "arguments": {"patch": patch}}))
        model = ScriptedChatModel(replies)
    else:
        env = {**read_env_file(args.env_file), **os.environ}
        model_name = env.get("LLM_MODEL", "").strip()
        config = AppConfig.for_model(model_name, workspace=workspace, state_dir=state_dir, user_id=user_id)
        model = HTTPChatModel(model_name, env.get("LLM_BASE_URL", ""), api_key_env="LLM_API_KEY",
                              api_key=env.get("LLM_API_KEY"), max_output_tokens=config.max_output_tokens,
                              json_output=model_name == "ecnu-max")
    automatic_tests = interactive and args.test_command is None
    command = DEFAULT_PROJECT_TEST_COMMAND if automatic_tests else tuple(shlex.split(args.test_command))
    regression = args.regression_command
    if args.sample:
        if command == ("python", "-m", "unittest", "discover", "-s", "tests"):
            command = ("python", "-m", "unittest", "tests.test_search.SearchTests.test_query_with_surrounding_spaces")
        regression = regression or "python -m unittest tests.test_search.SearchTests.test_plain_query"
    sandbox = DockerSandbox(args.image, command, args.timeout,
                            regression_command=tuple(shlex.split(regression)) if regression else None,
                            allowed_paths=tuple(args.allow_path), protected_tests=args.protected_tests,
                            expected_failure=args.expected_failure, scratch_dir=config.state_dir / "sandbox")
    dependencies = configure_retrieval(config, model, args.retrieval, args.env_file)
    if interactive:
        config.context.auto_summary_enabled = True
        dependencies["terminal_memory_dir"] = args.memory_dir
        dependencies["project_sandbox"] = DockerSandbox(
            args.image, tuple(shlex.split(args.project_test_command)), args.timeout,
            scratch_dir=config.state_dir / "sandbox",
        )
    return CodebaseMaintainer(
        config, model, sandbox=sandbox, index_pattern=args.index_pattern, carry_context=interactive,
        progress=display_progress if interactive else None, automatic_tests=automatic_tests, **dependencies,
    )


def run_repair(args: argparse.Namespace) -> int:
    agent = build_repair_agent(args)
    reply = agent.resume() if args.resume else agent.repair(args.issue or "修复查询首尾空格导致的漏匹配")
    print(reply.answer)
    if reply.repair is None:
        return 1
    result = reply.repair
    for phase, run in (reply.baseline or {}).items():
        print(f"修改前 {phase} {run['status']}，退出码 {run['exit_code']}")
    for phase, run in result.get("verification", {}).items():
        print(f"修改后 {phase} {run['status']}，退出码 {run['exit_code']}")
    for change in result.get("changed_files", []):
        print(f"{change['path']} +{change['added']} -{change['removed']}")
    retrieval = [{"stage": event.stage, **event.metadata}
                 for event in reply.trace if event.stage in {"memory_index", "rag_index"}]
    for event in retrieval:
        print(f"检索 {event['stage']} {json.dumps(event, ensure_ascii=False)}")
    print(f"状态 {result['status']}，退出码 {result['exit_code']}")
    if result["output"]:
        print(result["output"])
    if args.report_out:
        with args.report_out.open("x", encoding="utf-8") as file:
            json.dump(repair_report(reply), file, ensure_ascii=False, indent=2)
    if result["patch"]:
        if args.patch_out and result["status"] == "passed":
            with args.patch_out.open("x", encoding="utf-8") as file:
                file.write(result["patch"])
        else:
            print(result["patch"])
    if args.branch and result["status"] == "passed":
        print(f"补丁已暂存于 {deliver(agent.config.workspace, result, args.branch, args.worktree)}")
    return 0 if result["status"] == "passed" else 1


def run_bench_rag(args: argparse.Namespace) -> dict:
    from contextlib import ExitStack
    from dataclasses import replace
    from urllib.parse import urlsplit

    code_benchmark = args.command == "bench-rag"
    enhance = code_benchmark and (args.hyde or args.multi_query)
    progress = lambda message: print(message, file=sys.stderr, flush=True)
    with ExitStack() as stack:
        vectors = reranker = query_model = None
        if enhance and not args.embedding:
            raise ValueError("HyDE 和多查询需要 --embedding")
        if args.embedding or args.rerank:
            import httpx

            transport = stack.enter_context(httpx.Client(trust_env=not args.direct))
        if args.embedding:
            from .embeddings import EmbeddingConfig, HTTPEmbeddingClient
            from .rag.repoqa_vectors import CachedBenchmarkVectors

            config = EmbeddingConfig.from_env(args.env_file)
            if not config.enabled:
                raise ValueError("请在 .env 配置 EMBEDDING_BASE_URL、EMBEDDING_MODEL 和 EMBEDDING_API_KEY")
            if args.embedding_batch_size is not None:
                config = replace(config, batch_size=args.embedding_batch_size)
            if args.workers <= 0:
                raise ValueError("--workers 必须为正整数")
            vectors = CachedBenchmarkVectors(HTTPEmbeddingClient(config, transport), args.vector_cache,
                                             workers=args.workers)
        if enhance:
            env = {**read_env_file(args.env_file), **os.environ}
            query_model = HTTPChatModel(env.get("LLM_MODEL", "").strip(), env.get("LLM_BASE_URL", "").strip(),
                                        api_key_env="LLM_API_KEY", api_key=env.get("LLM_API_KEY", "").strip(),
                                        timeout_seconds=60, transport=transport, json_output=True)
        if args.rerank:
            from .rag.rerank import ECNUReranker, RerankConfig

            rerank_config = RerankConfig.from_env(args.env_file)
            if not rerank_config.api_key:
                env = {**read_env_file(args.env_file), **os.environ}
                target = urlsplit(rerank_config.base_url)
                if target.scheme == "https" and target.hostname == "chat.ecnu.edu.cn" and target.port in (None, 443):
                    for service in ("EMBEDDING", "LLM"):
                        service_url = urlsplit(env.get(f"{service}_BASE_URL", "").strip())
                        key = env.get(f"{service}_API_KEY", "").strip()
                        if key and service_url.scheme == "https" and service_url.hostname == target.hostname and service_url.port in (None, 443):
                            rerank_config = replace(rerank_config, api_key=key)
                            break
            if not rerank_config.api_key:
                raise ValueError("请在 .env 配置 RERANK_API_KEY")
            reranker = ECNUReranker(rerank_config, transport)
        if not code_benchmark:
            from .rag.scifact import evaluate_scifact

            return evaluate_scifact(args.dataset, args.k, vectors=vectors, reranker=reranker,
                                    candidates=args.candidates, rerank_candidates=args.rerank_candidates,
                                    progress=progress)
        from .rag.repoqa import evaluate_repoqa

        return evaluate_repoqa(args.dataset, args.k, vectors=vectors, reranker=reranker, candidates=args.candidates,
                               rerank_source=args.rerank_source, rerank_candidates=args.rerank_candidates,
                               query_model=query_model, hyde=args.hyde, multi_query=args.multi_query,
                               query_workers=args.query_workers, query_cache=args.query_cache, repo_limit=args.repo_limit,
                               source_coverage=args.source_coverage, progress=progress)


def _benchmark_options(parser: argparse.ArgumentParser, *, vector_cache: str, default_k: int) -> None:
    parser.add_argument("--k", type=int, default=default_k)
    parser.add_argument("--candidates", type=int, default=50, help="每路召回的候选上限")
    parser.add_argument("--embedding", action="store_true", help="使用 .env 配置的向量接口")
    parser.add_argument("--rerank", action="store_true", help="使用 .env 配置的重排接口")
    parser.add_argument("--rerank-candidates", type=int, help="只重排召回结果前 N 项")
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    parser.add_argument("--vector-cache", type=Path, default=PROJECT_ROOT / "runtime" / "benchmarks" / vector_cache)
    parser.add_argument("--embedding-batch-size", type=int, help="覆盖 .env 中的向量编码批大小")
    parser.add_argument("--workers", type=int, default=1, help="并行编码的批次数")
    parser.add_argument("--direct", action="store_true", help="直连 API，忽略系统代理环境变量")


def _repair_settings(repair: argparse.ArgumentParser, *, workspace_required: bool = False) -> None:
    repair.add_argument("--workspace", type=Path, required=workspace_required, help="目标仓库根目录")
    repair.add_argument("--state-dir", type=Path)
    repair.add_argument("--image", default="python:3.12-slim", help="预先下载或构建的 Docker 镜像")
    repair.add_argument("--test-command", default="python -m unittest discover -s tests",
                        help="指定沙箱测试命令。终端模式默认按现有 Python 项目自动发现")
    repair.add_argument("--regression-command", help="修复前后均需通过的独立回归命令")
    repair.add_argument("--expected-failure", help="复现失败日志必须包含的文本")
    repair.add_argument("--protected-tests", type=Path, help="只读挂载到容器 /agent-tests 的验收用例目录")
    repair.add_argument("--allow-path", action="append", default=[], help="允许修改的相对路径模式，可重复")
    repair.add_argument("--timeout", type=int, default=90, help="每次测试的超时秒数")
    repair.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    repair.add_argument("--retrieval", choices=("lexical", "vector", "hybrid"), default="lexical",
                        help="修改项目时使用的源码检索模式")
    repair.add_argument("--index-pattern", default="**/*.py", help="修复模式的源码检索范围，使用仓库相对 glob")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="第九章上下文工程练习骨架")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("overview", help="展示类初始化与组件关系，不执行核心 TODO")
    check = sub.add_parser("check", help="运行指定阶段验收；未完成练习应失败")
    check.add_argument("stage", choices=(*TEST_STAGES, "all"))
    example = sub.add_parser("example", help="执行某个初始化/调用示例，遇 TODO 时退出 2")
    example.add_argument("name", choices=tuple(EXAMPLES))
    demo = sub.add_parser("demo", help="尝试完整工作流，遇 TODO 时明确停止")
    demo.add_argument("--mode", choices=("explore", "issue", "plan"), default="explore")
    demo.add_argument("--question", default="为什么带空格的查询可能漏掉相关文档？")
    demo.add_argument("--workspace", type=Path)
    demo.add_argument("--state-dir", type=Path)
    demo.add_argument("--resume", action="store_true", help="恢复最近一次未结束的回合")
    repair = sub.add_parser("repair", help="生成候选补丁并在 Docker 中测试")
    _repair_settings(repair)
    repair.add_argument("--issue", help="要修复的问题")
    repair.add_argument("--report-out", type=Path, help="将测试与变更报告写入一个新的 JSON 文件")
    repair.add_argument("--branch", help="通过后创建的新分支名")
    repair.add_argument("--worktree", type=Path, help="新分支的独立工作树路径")
    repair.add_argument("--patch-out", type=Path, help="将通过测试的补丁写入一个新文件")
    mode = repair.add_mutually_exclusive_group()
    mode.add_argument("--sample", action="store_true", help="对自带的示例仓库演示一次失败后重试")
    mode.add_argument("--resume", action="store_true", help="继续使用同一仓库快照的修复回合")
    terminal = sub.add_parser("terminal", help="启动可持续提问、修复和恢复的交互终端")
    _repair_settings(terminal)
    terminal.add_argument("--project-test-command", default=shlex.join(DEFAULT_PROJECT_TEST_COMMAND),
                          help="新项目在镜像内的验收命令")
    terminal.add_argument("--new-session", action="store_true", help="开启新会话，加载同一用户和项目的长期记忆")
    terminal.add_argument("--no-stream", action="store_true", help="直接显示完整答复，跳过流式生成")
    terminal.add_argument("--memory-dir", type=Path, default=Path.home() / ".context-lab" / "memory",
                          help="长期记忆目录，可在不同会话状态目录之间共享")
    terminal.add_argument("--user-id", default="student", help="区分本机不同用户的会话和记忆")
    terminal.set_defaults(sample=False, workspace=Path("."), test_command=None)
    bench = sub.add_parser("bench-rag", help="用 RepoQA Python 子集评测 BM25、向量、混合召回和重排")
    bench.add_argument("dataset", type=Path, help="RepoQA 官方 JSON 或 JSON.GZ 文件")
    _benchmark_options(bench, vector_cache="repoqa-vectors.sqlite3", default_k=5)
    bench.add_argument("--rerank-source", choices=("hybrid", "vector", "enhanced"), default="hybrid",
                       help="选择混合、纯向量或查询增强候选进行重排")
    bench.add_argument("--hyde", action="store_true", help="对比原查询加假想文档的向量检索")
    bench.add_argument("--multi-query", action="store_true", help="对比原查询加最多三条改写的向量检索")
    bench.add_argument("--query-workers", type=int, default=1, help="并行生成查询的题数")
    bench.add_argument("--query-cache", type=Path, default=PROJECT_ROOT / "runtime" / "benchmarks" / "repoqa-queries.json",
                       help="缓存已生成且验证通过的查询计划")
    bench.add_argument("--source-coverage", action="store_true", help="评测 o200k 源码预算内的目标函数行覆盖率")
    bench.add_argument("--repo-limit", type=int, help="仅评测前几个仓库，便于验证接口")
    documents = sub.add_parser("bench-docs", help="用 BEIR SciFact 官方测试集评测文档检索")
    documents.add_argument("dataset", type=Path, help="BEIR 官方 scifact.zip 文件")
    _benchmark_options(documents, vector_cache="scifact-vectors.sqlite3", default_k=10)
    args = parser.parse_args(argv)
    if args.command == "terminal":
        try:
            return run_terminal(build_repair_agent(args, interactive=True), new_session=args.new_session,
                                stream=not args.no_stream)
        except (OSError, ValueError, RuntimeError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
    if args.command == "repair":
        if bool(args.branch) != bool(args.worktree):
            repair.error("--branch 和 --worktree 需要一起提供")
        if args.sample and args.workspace is not None:
            repair.error("--sample 使用内置工作区")
        if not args.sample and (args.workspace is None or not (args.issue or args.resume)):
            repair.error("请指定 --workspace 以及 --issue 或 --resume")
        try:
            return run_repair(args)
        except (OSError, ValueError, RuntimeError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
    if args.command in {"bench-rag", "bench-docs"}:
        try:
            print(json.dumps(run_bench_rag(args), ensure_ascii=False, indent=2))
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError) as exc:
            print(f"评测无法运行：{exc}", file=sys.stderr)
            return 2
        return 0
    if args.command == "example":
        from examples.common import run_learning_example
        module = importlib.import_module(f"examples.{EXAMPLES[args.name]}")
        run_learning_example(module.main)
        return 0
    if args.command == "check":
        if not (PROJECT_ROOT / "tests").is_dir():
            parser.error("未找到练习测试，请在完整源码项目中运行")
        sys.path.insert(0, str(PROJECT_ROOT))
        names = TEST_STAGES if args.stage == "all" else (args.stage,)
        suite = unittest.TestSuite(
            unittest.defaultTestLoader.loadTestsFromName(f"tests.test_{stage}")
            for stage in names
        )
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1
    agent = make_demo_agent(
        getattr(args, "workspace", None), getattr(args, "state_dir", None),
        sample_rag=args.command == "demo", resume=getattr(args, "resume", False),
    )
    if args.command == "overview":
        overview(agent)
        return 0
    try:
        if args.resume:
            reply = agent.resume()
        elif args.mode == "explore":
            reply = agent.explore()
        elif args.mode == "issue":
            reply = agent.analyze_issue(args.question)
        else:
            reply = agent.plan_task(args.question)
        print(reply.answer)
        print(f"上下文计数：{reply.context_result.token_count}")
        for event in reply.trace:
            if event.stage == "tool":
                print(f"工具：{event.metadata['tool']} ({event.metadata['status']})")
        for warning in reply.context_result.warnings:
            print(f"资料提示：{warning}")
        print("说明：默认是预设回复模型，不代表真实推理效果。")
        return 0
    except ExercisePending as exc:
        print(str(exc))
        print("这是初始练习稿的预期状态，请按 docs/学习路线.md 补齐后重跑。")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
