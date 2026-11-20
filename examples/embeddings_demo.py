"""共享增量索引示例：显式运行将连接数据库并请求嵌入服务。"""

from context_lab.embeddings import EmbeddingChunk, EmbeddingConfig, IncrementalEmbeddingIndex


def main():
    config = EmbeddingConfig.from_env()
    index = IncrementalEmbeddingIndex(config)
    print("配置已读取；嵌入是否配置完整：", config.enabled)
    print("PostgreSQL 地址是否已配置：", config.database_configured)
    print("构造服务不会建库、扫描文件或发送请求。")
    index.initialize()  # 显式连接 PostgreSQL 并初始化存储结构。

    namespace = "code:demo-repo"
    source_id = "src/cache.py"
    chunks = [
        EmbeddingChunk("function:get", "Read an item from cache", {"symbol": "get"}),
        EmbeddingChunk("function:put", "Write an item into cache", {"symbol": "put"}),
    ]
    print(index.sync_source(namespace, source_id, chunks))  # 只记录变更并排队
    print(index.status(namespace))
    print("索引成功提交块数：", index.process_pending())  # 单独执行入库向量化
    print(index.search("read cache", namespace))  # 只给 query 生成向量

    # 修改第二段：应只生成新文本的向量，第一段复用原缓存。
    chunks[1].text = "Write an item into cache with TTL"
    print(index.sync_source(namespace, source_id, chunks))
    print("本次提交块数：", index.process_pending())


if __name__ == "__main__":
    main()
