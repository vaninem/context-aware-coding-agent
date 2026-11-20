"""管理项目内的 PostgreSQL 验收实例；不注册登录启动服务，不打印连接口令。"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from context_lab.embeddings import EmbeddingConfig
from context_lab.storage.env_file import read_env_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "stop", "status"), nargs="?", default="start")
    args = parser.parse_args()
    binary = os.environ.get("PG_BIN") or shutil.which("pg_ctl")
    pg_bin = Path(binary).parent if binary and Path(binary).name == "pg_ctl" else Path(binary or "/opt/homebrew/opt/postgresql@16/bin")
    home = ROOT / "runtime" / "postgres"
    data = home / "data"
    pg_ctl = str(pg_bin / "pg_ctl")
    if not (pg_bin / "pg_ctl").is_file():
        raise RuntimeError("未找到 PostgreSQL 16；请安装后设置 PG_BIN 为其 bin 目录")

    running = data.is_dir() and subprocess.run(
        [pg_ctl, "-D", str(data), "status"], capture_output=True,
    ).returncode == 0
    if args.action == "status":
        print("项目 PostgreSQL 正在运行。" if running else "项目 PostgreSQL 未运行。")
        return 0 if running else 1
    if args.action == "stop":
        if running:
            subprocess.run([pg_ctl, "-D", str(data), "-m", "fast", "-w", "stop"], check=True, capture_output=True)
        print("项目 PostgreSQL 已停止。")
        return 0

    config = EmbeddingConfig.from_env()
    info = conninfo_to_dict(config.database_url)
    if info.get("host") not in ("127.0.0.1", "localhost"):
        raise ValueError("本地实例只接受 DATABASE_URL 指向 127.0.0.1 或 localhost")
    role, password, database = info.get("user"), info.get("password"), info.get("dbname")
    if not role or not password or not database:
        raise ValueError("DATABASE_URL 必须显式包含用户名、口令和数据库名")
    port = int(info.get("port", 5432))
    if not 1024 <= port <= 65535:
        raise ValueError("本地数据库端口必须在 1024 至 65535 之间")
    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    socket_dir = Path("/private/tmp") / ("context-lab-pg-" + hashlib.sha256(str(ROOT).encode()).hexdigest()[:10])
    socket_dir.mkdir(mode=0o700, exist_ok=True)
    if not (data / "PG_VERSION").exists():
        password_file = home / "init-password"
        descriptor = os.open(password_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "w") as file:
                file.write(password + "\n")
            subprocess.run([
                str(pg_bin / "initdb"), "-D", str(data), "--username=" + role,
                "--auth-local=trust", "--auth-host=scram-sha-256", "--pwfile=" + str(password_file),
                "--encoding=UTF8", "--locale=C",
            ], check=True, capture_output=True)
        finally:
            password_file.unlink(missing_ok=True)
    if not running:
        options = f"-h 127.0.0.1 -p {port} -k {socket_dir} -c shared_buffers=32MB -c max_connections=30"
        subprocess.run([pg_ctl, "-D", str(data), "-l", str(home / "server.log"), "-o", options, "-w", "start"],
                       check=True, capture_output=True)

    with psycopg.connect(make_conninfo(config.database_url, dbname="postgres"), autocommit=True, connect_timeout=5) as conn:
        for name in dict.fromkeys((database, "context_lab_test")):
            if conn.execute("SELECT 1 FROM pg_database WHERE datname=%s", (name,)).fetchone() is None:
                conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    test_url = make_conninfo(config.database_url, dbname="context_lab_test")
    for url in (config.database_url, test_url):
        with psycopg.connect(url, autocommit=True, connect_timeout=5) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

    test_file = ROOT / ".env.test"
    existing = read_env_file(test_file).get("TEST_DATABASE_URL")
    if existing and existing != test_url:
        raise RuntimeError("已有 .env.test 指向另一测试库，已保留该文件")
    if not existing:
        descriptor = os.open(test_file, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(descriptor, "a") as file:
            file.write("\n# 项目本地验收库；不要提交此文件。\nTEST_DATABASE_URL=" + json.dumps(test_url) + "\n")
    test_file.chmod(0o600)
    print("项目 PostgreSQL 已启动；项目库和专用验收库均已启用 pgvector。")
    print("数据目录：runtime/postgres/data；验收连接配置：.env.test（权限 0600）。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"本地 PostgreSQL 操作失败（{type(exc).__name__}）；请检查安装与 runtime/postgres/server.log。")
        raise SystemExit(1) from None
