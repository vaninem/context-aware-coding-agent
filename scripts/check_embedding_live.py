"""显式运行：用少量合成代码验证真实模型与专用测试库，结束后清理测试 schema。"""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from context_lab.embeddings import EmbeddingChunk, EmbeddingConfig, IncrementalEmbeddingIndex
from context_lab.storage.env_file import read_env_file
from context_lab.embeddings.errors import EmbeddingRequestError

NAMESPACE = "code:live-smoke"
SOURCE = "src/cache_example.py"
QUERY = "查找读取缓存的函数：数据不存在或者已经过期时返回 None，未过期时返回缓存值。"
READ = '''def get_valid_cache(cache, key, now):
    """按 key 读取缓存：不存在或已过期返回 None，未过期返回 value。"""
    entry = cache.get(key)
    if entry is None or entry["expires_at"] <= now:
        return None
    return entry["value"]
'''
WRITE = '''def put_cache(cache, key, value):
    """向缓存中写入一个键值对。"""
    cache[key] = {"value": value, "expires_at": float("inf")}
'''
WRITE_TTL = '''def put_cache(cache, key, value, now, ttl_seconds):
    """写入缓存，并根据当前时间和 TTL 设置过期时间。"""
    cache[key] = {"value": value, "expires_at": now + ttl_seconds}
'''


def chunks(modified=False):
    return [EmbeddingChunk("get_valid_cache", READ), EmbeddingChunk("put_cache", WRITE_TTL if modified else WRITE)]


def configuration(schema):
    settings = read_env_file(ROOT / ".env.test")
    url = os.environ.get("TEST_DATABASE_URL", settings.get("TEST_DATABASE_URL", ""))
    if not url:
        raise RuntimeError("需要专用 TEST_DATABASE_URL，不回退业务库")
    config = EmbeddingConfig.from_env()
    if not config.enabled:
        raise RuntimeError("真实嵌入服务尚未配置")
    return replace(config, database_url=url, database_schema=schema, batch_size=2)


def verify_child(schema):
    index = IncrementalEmbeddingIndex(configuration(schema))
    index.initialize()
    unchanged = index.sync_source(NAMESPACE, SOURCE, chunks(modified=True))
    assert unchanged.unchanged == 2 and unchanged.queued == 0
    assert index.process_pending() == 0
    copied = index.sync_source(NAMESPACE, "src/copied.py", [EmbeddingChunk("copied_read", READ)])
    assert copied.cache_hits == 1 and copied.queued == 0
    assert index.process_pending() == 0
    assert index.client.calls == 0
    return {"unchanged": unchanged.unchanged, "queued": unchanged.queued, "cache_hits": copied.cache_hits,
            "ready": index.status(NAMESPACE).ready, "model_requests": index.client.calls}


def process_live(index, report, stage):
    """按现有队列策略等待有限重试，不把临时接口故障伪装成成功。"""
    completed = 0
    for attempt in range(index.config.max_attempts):
        completed += index.process_pending()
        if index.last_process_report.request_errors:
            report["observed_request_errors"].append({"stage": stage, "batches": index.last_process_report.request_errors})
        status = index.status(NAMESPACE)
        if not status.pending and not status.running and not status.failed:
            return completed
        if status.failed or attempt + 1 == index.config.max_attempts:
            raise RuntimeError("真实索引任务未完成")
        time.sleep(5.1)
    return completed


def search_live(index, report):
    for attempt in range(2):
        try:
            return index.search(QUERY, NAMESPACE, limit=2)
        except EmbeddingRequestError as exc:
            report["observed_request_errors"].append({"stage": "query", "code": exc.error_code})
            if not exc.retryable or attempt == 1:
                raise
            time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restart-db", action="store_true", help="同时重启本项目管理的本地数据库以验证落盘")
    parser.add_argument("--verify-child", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.verify_child:
        if not args.verify_child.startswith("test_live_"):
            raise ValueError("仅接受测试 schema")
        print(json.dumps(verify_child(args.verify_child)))
        return 0

    schema = "test_live_" + uuid4().hex
    config = configuration(schema)
    index = IncrementalEmbeddingIndex(config)
    report = {"started_at": datetime.now(timezone.utc).isoformat(), "ok": False,
              "postgres_restarted": False, "uses_real_embedding": True, "observed_request_errors": []}
    output = ROOT / "runtime" / "embedding_live_smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        index.initialize()
        with index.store.connect() as conn:
            report["postgres_version"] = conn.execute("SHOW server_version").fetchone()[0]
            report["pgvector_version"] = conn.execute("SELECT extversion FROM pg_extension WHERE extname='vector'").fetchone()[0]
        initial = index.sync_source(NAMESPACE, SOURCE, chunks())
        assert initial.queued == 2 and initial.created == 2
        report["initial_sync"] = asdict(initial)
        assert process_live(index, report, "initial") == 2
        report["initial_process"] = asdict(index.last_process_report)
        report["initial_requests"] = index.client.calls
        hits = search_live(index, report)
        assert hits and hits[0].chunk.chunk_id == "get_valid_cache"
        report["query_top_id"] = hits[0].chunk.chunk_id
        report["query_top_score"] = hits[0].score
        old_write_hash = index._content_hash(WRITE)
        before = index.client.calls
        changed = index.sync_source(NAMESPACE, SOURCE, chunks(modified=True))
        assert changed.updated == 1 and changed.unchanged == 1 and changed.queued == 1
        assert process_live(index, report, "changed") == 1
        report["changed_requests"] = index.client.calls - before
        report["changed_sync"] = asdict(changed)
        report["changed_process"] = asdict(index.last_process_report)
        assert index.store.get_cached(index.profile_id, old_write_hash) is not None
        with index.store.connect() as conn:
            report["dimensions"] = conn.execute(sql.SQL("SELECT dimensions FROM {} WHERE profile_id=%s")
                                                .format(sql.Identifier(schema, "profiles")), (index.profile_id,)).fetchone()[0]

        if args.restart_db:
            main_info = conninfo_to_dict(EmbeddingConfig.from_env().database_url)
            test_info = conninfo_to_dict(config.database_url)
            assert test_info.get("host") in ("127.0.0.1", "localhost")
            assert main_info.get("port") == test_info.get("port")
            for action in ("stop", "start"):
                subprocess.run([sys.executable, str(ROOT / "scripts" / "local_postgres.py"), action],
                               check=True, capture_output=True)
            report["postgres_restarted"] = True
        child = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--verify-child", schema],
                               check=True, capture_output=True, text=True)
        report["new_process"] = json.loads(child.stdout)
        assert report["new_process"]["ready"] == 3

        deleted = index.delete_source(NAMESPACE, SOURCE)
        assert deleted.deleted == 2
        assert index.store.list_source(index.profile_id, NAMESPACE, SOURCE) == []
        cached_read = index.store.get_cached(index.profile_id, index._content_hash(READ))
        assert index.store.search_ready(index.profile_id, NAMESPACE, cached_read, source_id=SOURCE) == []
        hits = index.store.search_ready(index.profile_id, NAMESPACE, cached_read, source_id="src/copied.py")
        assert len(hits) == 1 and hits[0].chunk.chunk_id == "copied_read"
        report["deleted_source_chunks"] = deleted.deleted
        report["other_source_preserved"] = True
        index.delete_source(NAMESPACE, "src/copied.py")
        before = index.client.calls
        assert index.search(QUERY, NAMESPACE) == []
        assert index.client.calls == before
        report["empty_search_requests"] = 0
        report["main_process_model_requests"] = index.client.calls
        report["ok"] = True
    except Exception as exc:
        report["error_type"] = type(exc).__name__
        report["error_code"] = getattr(exc, "error_code", "unavailable")
    finally:
        try:
            with psycopg.connect(config.database_url) as conn:
                conn.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema)))
            report["test_schema_cleaned"] = True
        except Exception:
            report["test_schema_cleaned"] = False
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] and report["test_schema_cleaned"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"ok": False, "error_type": type(exc).__name__}))
        raise SystemExit(1) from None
