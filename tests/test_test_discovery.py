from context_lab.tools.test_discovery import discover_tests
from tests.support import SandboxTestCase


class TestDiscoveryTests(SandboxTestCase):
    def test_unittest_recognizes_a_suite_without_pytest_config(self):
        target = self.tmp / "tests"
        target.mkdir()
        (target / "test_value.py").write_text(
            "from unittest import TestCase\nclass ValueTest(TestCase):\n    def test_value(self): pass\n",
            encoding="utf-8",
        )
        self.assertEqual(discover_tests(self.tmp), {
            "command": ["python", "-m", "unittest", "discover", "-s", "tests"], "source": "tests",
        })

    def test_pytest_discovers_standalone_functions_or_configuration(self):
        source = self.tmp / "value_test.py"
        source.write_text("def test_value():\n    assert True\n", encoding="utf-8")
        self.assertEqual(discover_tests(self.tmp)["command"], ["python", "-m", "pytest", "-q"])
        source.unlink()
        (self.tmp / "pyproject.toml").write_text("[tool.pytest.ini_options]\naddopts = '-x'\n", encoding="utf-8")
        self.assertEqual(discover_tests(self.tmp)["source"], "pyproject.toml")

    def test_no_suite_does_not_use_a_comment_or_a_virtualenv(self):
        (self.tmp / "test_helper.py").write_text("# helper only\n", encoding="utf-8")
        environment = self.tmp / ".venv"
        environment.mkdir()
        (environment / "test_used_by_dependency.py").write_text("def test_example(): pass\n", encoding="utf-8")
        self.assertEqual(discover_tests(self.tmp)["command"], [])
