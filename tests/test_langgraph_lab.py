"""LG01 至 LG05 的独立验收，使用本地模型回复与工具替身。"""

import json
import unittest
from unittest.mock import Mock

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END

from context_lab.models import ScriptedChatModel
from examples.langgraph_lab import EvidenceWorkflow, resume_turn, start_turn


class LangGraphLabTests(unittest.TestCase):
    def make_workflow(self, replies=(), *, max_tool_calls=3):
        self.model = ScriptedChatModel(replies)
        self.search = Mock(return_value="src/search.py:3")
        self.read_file = Mock(return_value="def normalize_query(query): return query.lower()")
        return EvidenceWorkflow(self.model, self.search, self.read_file, max_tool_calls=max_tool_calls)

    def test_lg01_routes_final_answer_and_each_tool_request(self):
        workflow = self.make_workflow()
        state = {"action": None, "require_review": True}
        self.assertEqual(workflow.route_after_decision(state), END)
        state["action"] = {"tool": "search", "argument": "normalize_query"}
        self.assertEqual(workflow.route_after_decision(state), "execute_tool")
        state["action"] = {"tool": "read_file", "argument": "src/search.py"}
        self.assertEqual(workflow.route_after_decision(state), "review_action")
        state["require_review"] = False
        self.assertEqual(workflow.route_after_decision(state), "execute_tool")

    def test_lg02_graph_loops_through_model_and_tools(self):
        workflow = self.make_workflow([
            '{"tool":"search","argument":"normalize_query"}',
            '{"tool":"read_file","argument":"src/search.py"}',
            '{"answer":"当前实现只转小写。"}',
        ])
        graph = workflow.build_graph()
        result = graph.invoke({
            "query": "空格为什么会影响查询", "require_review": False, "observations": [],
            "tool_calls": 0, "action": None, "answer": "",
        })
        self.search.assert_called_once_with("normalize_query")
        self.read_file.assert_called_once_with("src/search.py")
        self.assertEqual(result["tool_calls"], 2)
        self.assertEqual(result["answer"], "当前实现只转小写。")
        seen = json.loads(self.model.calls[-1][1].content)["observations"]
        self.assertEqual(seen, result["observations"])

    def test_lg02_tool_budget_ends_the_loop(self):
        workflow = self.make_workflow(['{"tool":"search","argument":"q"}'] * 2, max_tool_calls=1)
        graph = workflow.build_graph()
        result = graph.invoke({
            "query": "q", "require_review": False, "observations": [],
            "tool_calls": 0, "action": None, "answer": "",
        })
        self.search.assert_called_once_with("q")
        self.assertEqual(result["tool_calls"], 1)
        self.assertEqual(result["answer"], "已达到取证上限。")

    def test_lg03_start_turn_initializes_state_and_passes_thread_id(self):
        graph = Mock()
        result = start_turn(graph, "定位查询函数", thread_id="issue-1", require_review=True)
        graph.invoke.assert_called_once()
        self.assertIs(result, graph.invoke.return_value)
        args, kwargs = graph.invoke.call_args
        state = args[0] if args else kwargs["input"]
        config = args[1] if len(args) > 1 else kwargs["config"]
        self.assertEqual(config["configurable"]["thread_id"], "issue-1")
        self.assertEqual(state, {
            "query": "定位查询函数", "require_review": True, "observations": [],
            "tool_calls": 0, "action": None, "answer": "",
        })

    def test_lg04_pauses_before_read_and_returns_the_requested_action(self):
        workflow = self.make_workflow(['{"tool":"read_file","argument":"src/search.py"}'])
        graph = workflow.build_graph(InMemorySaver())
        result = start_turn(graph, "查看文件", thread_id="pause", require_review=True)
        self.read_file.assert_not_called()
        self.assertEqual(result["__interrupt__"][0].value, {"tool": "read_file", "argument": "src/search.py"})

    def test_lg05_approval_continues_and_rejection_skips_the_tool(self):
        for approved in (True, False):
            with self.subTest(approved=approved):
                workflow = self.make_workflow([
                    '{"tool":"read_file","argument":"src/search.py"}', '{"answer":"取证完成"}',
                ])
                graph = workflow.build_graph(InMemorySaver())
                start_turn(graph, "查看文件", thread_id="review", require_review=True)
                result = resume_turn(graph, thread_id="review", approved=approved)
                if approved:
                    self.read_file.assert_called_once_with("src/search.py")
                    self.assertEqual(result["answer"], "取证完成")
                    self.assertEqual(len(self.model.calls), 2)
                else:
                    self.read_file.assert_not_called()
                    self.assertEqual(result["answer"], "用户拒绝读取文件，已结束取证。")
                    self.assertEqual(len(self.model.calls), 1)
                self.assertIsNone(result["action"])
