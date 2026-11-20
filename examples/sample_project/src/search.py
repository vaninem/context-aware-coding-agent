"""供只读工具探索的合成样例，尚未处理查询的首尾空白。"""


def normalize_query(query: str) -> str:
    return query.lower()


def search_documents(query: str, documents: list[str]) -> list[str]:
    normalized = normalize_query(query)
    return [document for document in documents if normalized in document.lower()]
