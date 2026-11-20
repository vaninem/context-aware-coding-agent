"""显式读取测试库地址；绝不回退 DATABASE_URL，不请求真实嵌入服务。"""

import os
from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from context_lab.storage.env_file import read_env_file


def main() -> int:
    settings = {**read_env_file(PROJECT_ROOT / ".env"), **read_env_file(PROJECT_ROOT / ".env.test")}
    test_url = os.environ.get("TEST_DATABASE_URL", settings.get("TEST_DATABASE_URL", ""))
    if not test_url.strip():
        print("请先在本地 .env 或环境变量中显式配置专用 TEST_DATABASE_URL。")
        return 2
    os.environ["TEST_DATABASE_URL"] = test_url.strip()
    suite = unittest.defaultTestLoader.discover(
        start_dir=str(PROJECT_ROOT / "tests"), pattern="test_embedding*.py", top_level_dir=str(PROJECT_ROOT),
    )
    suite.addTests(unittest.defaultTestLoader.loadTestsFromName("tests.test_scaffold"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
