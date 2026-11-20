"""有限的离线测试替身和 Chat Completions 服务适配器。"""

import json
import os
from copy import deepcopy
from http.client import HTTPException
from typing import Generator, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .errors import ModelNotConfigured
from .types import Message


class ModelRequestError(RuntimeError):
    """脱敏的模型请求错误，可由调用方决定是否重试。"""

    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class ScriptedChatModel:
    """按顺序返回预先写好的文字，仅用于检查编排，不具有推理能力。"""

    def __init__(self, replies: Iterable[str]) -> None:
        self.replies = list(replies)
        self.calls: list[list[Message]] = []
        self._cursor = 0

    def generate(self, messages: list[Message]) -> str:
        if self._cursor >= len(self.replies):
            raise ModelNotConfigured("离线测试回复已用完；请提供更多 replies 或接入真实模型")
        self.calls.append(deepcopy(messages))
        response = self.replies[self._cursor]
        self._cursor += 1
        return response


class HTTPChatModel:
    """调用兼容 Chat Completions 的文本与流式接口。"""

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key_env: str = "CHAT_API_KEY",
        timeout_seconds: float = 30.0,
        *,
        api_key: str | None = None,
        transport=None,
        json_output: bool = False,
        max_output_tokens: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self._api_key = api_key
        self.transport = transport
        self.json_output = json_output
        self.max_output_tokens = max_output_tokens
        self.timeout_seconds = timeout_seconds
        self.calls = 0
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds 必须大于 0")
        if max_output_tokens is not None and (type(max_output_tokens) is not int or max_output_tokens <= 0):
            raise ValueError("max_output_tokens 必须为正整数")

    def _request_data(self, messages: list[Message], *, stream: bool) -> tuple[str, dict, dict]:
        key = self._api_key or os.getenv(self.api_key_env)
        if not key or not self.model_name or not self.base_url.startswith(("https://", "http://")):
            raise ModelNotConfigured("请配置模型名、API 地址和密钥")
        body = {
            "model": self.model_name, "stream": stream,
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
        }
        if self.json_output and not stream:
            body["response_format"] = {"type": "json_object"}
        if self.max_output_tokens is not None:
            body["max_tokens"] = self.max_output_tokens
        headers = {
            "Authorization": f"Bearer {key}", "Content-Type": "application/json",
        }
        if stream:
            headers["Accept"] = "text/event-stream"
        return f"{self.base_url}/chat/completions", headers, body

    def generate(self, messages: list[Message]) -> str:
        """发送标准消息列表，从第一个候选中取出文本。"""
        endpoint, headers, body = self._request_data(messages, stream=False)
        try:
            self.calls += 1
            if self.transport is None:
                request = Request(endpoint, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    data = json.load(response)
            else:
                import httpx

                try:
                    response = self.transport.post(endpoint, json=body, headers=headers, timeout=self.timeout_seconds)
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    code = exc.response.status_code
                    raise ModelRequestError(f"模型服务返回 HTTP {code}", retryable=code in (408, 429) or code >= 500) from None
                except httpx.RequestError as exc:
                    raise ModelRequestError(f"模型服务连接失败（{type(exc).__name__}）", retryable=True) from None
                data = response.json()
            answer = data["choices"][0]["message"]["content"]
        except HTTPError as exc:
            raise ModelRequestError(f"模型服务返回 HTTP {exc.code}", retryable=exc.code in (408, 429) or exc.code >= 500) from None
        except (URLError, TimeoutError):
            raise ModelRequestError("模型服务连接失败或超时", retryable=True) from None
        except (ValueError, TypeError, KeyError, IndexError):
            raise RuntimeError("模型服务响应格式错误") from None
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("模型服务没有返回文本")
        return answer

    def generate_stream(self, messages: list[Message]) -> Generator[str, None, None]:
        """在响应尚未结束时产出正文片段，关闭生成器会关闭 HTTP 响应。"""
        endpoint, headers, body = self._request_data(messages, stream=True)
        try:
            self.calls += 1
            if self.transport is None:
                request = Request(endpoint, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    yield from self._stream_text(response)
            else:
                import httpx

                try:
                    with self.transport.stream("POST", endpoint, json=body, headers=headers,
                                               timeout=self.timeout_seconds) as response:
                        response.raise_for_status()
                        yield from self._stream_text(response.iter_lines())
                except httpx.HTTPStatusError as exc:
                    code = exc.response.status_code
                    raise ModelRequestError(f"模型服务返回 HTTP {code}", retryable=code in (408, 429) or code >= 500) from None
                except httpx.RequestError as exc:
                    raise ModelRequestError(f"正文连接中断（{type(exc).__name__}）", retryable=True) from None
        except HTTPError as exc:
            raise ModelRequestError(f"模型服务返回 HTTP {exc.code}", retryable=exc.code in (408, 429) or exc.code >= 500) from None
        except (URLError, OSError, HTTPException):
            raise ModelRequestError("正文连接中断或超时，可用 /resume 继续", retryable=True) from None
        except (ValueError, TypeError, KeyError, AttributeError):
            raise ModelRequestError("模型服务返回了无效的正文流") from None

    @staticmethod
    def _stream_text(lines: Iterable[bytes | str]) -> Generator[str, None, None]:
        """按 SSE 的空行分隔事件，只读第一个候选中的正文增量。"""
        data: list[str] = []
        for raw in lines:
            line = (raw.decode("utf-8") if isinstance(raw, bytes) else raw).rstrip("\r\n")
            if line.startswith("data:"):
                data.append(line[5:].removeprefix(" "))
                continue
            if line or not data:
                continue
            event = "\n".join(data)
            data.clear()
            if event.strip() == "[DONE]":
                return
            payload = json.loads(event)
            if "error" in payload:
                raise ModelRequestError("模型服务在生成正文时报告错误")
            for choice in payload.get("choices", []):
                if choice.get("index", 0) != 0:
                    continue
                delta = (choice.get("delta") or {}).get("content")
                if isinstance(delta, str) and delta:
                    yield delta
                if choice.get("finish_reason") == "length":
                    raise ModelRequestError("正文达到输出上限，可用 --no-stream 和 /resume 显示已保存的答复要点")
                if choice.get("finish_reason") == "stop":
                    return
        raise ModelRequestError("正文流在结束前断开，可用 /resume 继续", retryable=True)
