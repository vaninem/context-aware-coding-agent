"""E11：模拟网络验证批处理和错误处理，不读取 .env、不请求真实接口。"""

from dataclasses import replace
import json
import unittest

import httpx

from context_lab.embeddings import EmbeddingConfig, HTTPEmbeddingClient
from context_lab.embeddings.errors import EmbeddingNotConfigured, EmbeddingRequestError


class EmbeddingClientTests(unittest.TestCase):
    def setUp(self):
        self.config = EmbeddingConfig(
            base_url="https://invalid.example/v1", model="test-model", api_key="fake-test-key",
            batch_size=2, expected_dimensions=2,
        )

    def client_for(self, handler, config=None):
        http = httpx.Client(transport=httpx.MockTransport(handler))
        self.addCleanup(http.close)
        return HTTPEmbeddingClient(config or self.config, transport=http), http

    def test_batches_reorder_and_flatten_without_closing_injected_client(self):
        batches = []
        def handler(request):
            self.assertEqual(str(request.url), "https://invalid.example/v1/embeddings")
            self.assertEqual(request.method, "POST")
            self.assertEqual(request.headers["Authorization"], "Bearer fake-test-key")
            body = json.loads(request.content)
            self.assertEqual(body["model"], "test-model")
            self.assertEqual(body["encoding_format"], "float")
            self.assertEqual(request.extensions["timeout"]["read"], self.config.timeout_seconds)
            batches.append(body["input"])
            return httpx.Response(200, json={"data": [
                {"index": i, "embedding": [int(text), 1]}
                for i, text in reversed(list(enumerate(body["input"])))
            ]})
        client, http = self.client_for(handler)
        self.assertEqual(client.embed(["1", "2", "3", "4", "5"]),
                         [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0], [5.0, 1.0]])
        self.assertEqual(batches, [["1", "2"], ["3", "4"], ["5"]])
        self.assertEqual(client.calls, 3)
        self.assertFalse(http.is_closed)

    def test_empty_input_does_not_require_config_or_request(self):
        client = HTTPEmbeddingClient(EmbeddingConfig())
        self.assertEqual(client.embed([]), [])
        self.assertEqual(client.calls, 0)

    def test_missing_config_stops_before_request(self):
        for change in ({"api_key": ""}, {"model": ""}, {"base_url": ""}):
            client = HTTPEmbeddingClient(replace(self.config, **change))
            with self.subTest(change=change), self.assertRaises(EmbeddingNotConfigured):
                client.embed(["text"])
            self.assertEqual(client.calls, 0)

    def test_invalid_input_stops_before_request(self):
        client = HTTPEmbeddingClient(self.config)
        for texts in ("text", None, [1], [""], [" \n"]):
            with self.subTest(texts=texts), self.assertRaises(ValueError):
                client.embed(texts)
        self.assertEqual(client.calls, 0)

    def test_http_errors_are_distinct_and_redacted(self):
        for status, reason in ((400, "HTTP 请求失败"), (401, "鉴权失败"), (403, "无访问权限"),
                               (408, "HTTP 请求失败"), (429, "限流"), (500, "HTTP 请求失败"), (503, "HTTP 请求失败")):
            client, _ = self.client_for(lambda request: httpx.Response(status, text="fake-test-key sensitive-body"))
            with self.subTest(status=status), self.assertRaises(EmbeddingRequestError) as error:
                client.embed(["text"])
            message = str(error.exception)
            self.assertIn(reason, message)
            self.assertIn(str(status), message)
            self.assertNotIn("fake-test-key", message)
            self.assertNotIn("sensitive-body", message)
            self.assertEqual(client.calls, 1)
            self.assertEqual(error.exception.error_code, f"http_{status}")
            self.assertEqual(error.exception.retryable, status in (408, 429) or status >= 500)

    def test_timeout_and_network_errors_are_redacted(self):
        for kind, message in ((httpx.ReadTimeout, "超时"), (httpx.ConnectError, "网络连接失败")):
            def handler(request):
                raise kind("fake-test-key", request=request)
            client, _ = self.client_for(handler)
            with self.subTest(kind=kind), self.assertRaises(EmbeddingRequestError) as error:
                client.embed(["text"])
            self.assertIn(message, str(error.exception))
            self.assertNotIn("fake-test-key", str(error.exception))
            self.assertTrue(error.exception.retryable)
            self.assertEqual(error.exception.error_code, "timeout" if kind is httpx.ReadTimeout else "connection_error")

    def test_invalid_json_is_not_exposed(self):
        client, _ = self.client_for(lambda request: httpx.Response(200, text="fake-test-key"))
        with self.assertRaisesRegex(EmbeddingRequestError, "有效 JSON") as error:
            client.embed(["text"])
        self.assertEqual(error.exception.error_code, "invalid_json")
        self.assertFalse(error.exception.retryable)

    def test_invalid_embedding_response_is_rejected(self):
        client, _ = self.client_for(lambda request: httpx.Response(200, json={"data": []}))
        with self.assertRaises(EmbeddingRequestError):
            client.embed(["text"])

    def test_second_batch_failure_does_not_return_partial_results(self):
        batches = []
        def handler(request):
            batches.append(json.loads(request.content)["input"])
            if len(batches) == 2:
                return httpx.Response(429)
            return httpx.Response(200, json={"data": [
                {"index": 0, "embedding": [1, 1]}, {"index": 1, "embedding": [2, 1]},
            ]})
        client, _ = self.client_for(handler)
        with self.assertRaises(EmbeddingRequestError):
            client.embed(["1", "2", "3", "4", "5"])
        self.assertEqual(client.calls, 2)
        self.assertEqual(len(batches), 2)
