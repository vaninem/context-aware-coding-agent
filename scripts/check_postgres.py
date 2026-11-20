"""只检查连接和 vector 扩展，不创建业务表，不调用嵌入模型。"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from context_lab.embeddings import EmbeddingConfig


def main() -> int:
    config = EmbeddingConfig.from_env()
    if not config.database_configured:
        print("尚未配置 DATABASE_URL。")
        return 2
    try:
        import psycopg
    except ImportError:
        print("尚未安装 PostgreSQL 驱动；请在虚拟环境安装项目的 embedding 可选依赖。")
        return 2
    try:
        with psycopg.connect(config.database_url, connect_timeout=config.database_connect_timeout) as connection:
            extension = connection.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
            if extension is None:
                print("数据库连接成功，但尚未启用 vector 扩展。")
                return 1
            distance = connection.execute("SELECT '[1,0,0]'::vector <=> '[1,0,0]'::vector").fetchone()[0]
            if abs(float(distance)) > 1e-6:
                print("向量运算检查失败。")
                return 1
            print("PostgreSQL 连接成功；pgvector 扩展及余弦运算可用。")
            print("此脚本只检查连接和扩展；业务验收请运行 check_embedding_exercises.py。")
            return 0
    except Exception as exc:
        print(f"数据库检查失败（{type(exc).__name__}）；请检查服务、地址和权限。")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
