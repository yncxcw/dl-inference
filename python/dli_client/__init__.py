"""Dependency-free client for OpenAI-compatible servers such as `dli_serve`.

```python
from dli_client import Client

client = Client(base_url="http://127.0.0.1:8000/v1")
reply = client.chat.completions.create(
    model="Qwen/Qwen3.5-0.8B",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(reply.choices[0].message.content)

for chunk in client.chat.completions.create(model=..., messages=..., stream=True):
    if chunk.choices:
        print(chunk.choices[0].delta.content or "", end="", flush=True)
```

The interface follows the official `openai` package, which also works against
the same server.
"""

from __future__ import annotations

import http.client
import json
import urllib.error
import urllib.request
from typing import Any, Iterator, Mapping, Sequence

__all__ = [
    "APIConnectionError",
    "APIError",
    "APIObject",
    "APIStatusError",
    "Client",
    "Stream",
]


class APIError(Exception):
    """Base class for client errors."""


class APIConnectionError(APIError):
    """The server could not be reached, or the connection broke mid-response."""


class APIStatusError(APIError):
    """The server answered with an error."""

    def __init__(self, status_code: int, message: str, body: Any = None) -> None:
        super().__init__(f"HTTP {status_code}: {message}")
        self.status_code = status_code
        self.message = message
        self.body = body


class APIObject:
    """Read-only attribute view of a JSON object, like the OpenAI SDK's models.

    Missing fields read as None, as optional SDK fields do. `to_dict()` returns
    a copy of the underlying JSON.
    """

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, Any]) -> None:
        object.__setattr__(self, "_data", dict(data))

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return _wrap(self._data.get(name))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("API objects are read-only")

    def __getitem__(self, key: str) -> Any:
        return _wrap(self._data[key])

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __eq__(self, other: object) -> bool:
        return isinstance(other, APIObject) and self._data == other._data

    def __repr__(self) -> str:
        return f"APIObject({self._data!r})"

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._data))


def _wrap(value: Any) -> Any:
    if isinstance(value, Mapping):
        return APIObject(value)
    if isinstance(value, list):
        return [_wrap(item) for item in value]
    return value


class Stream:
    """Iterates server-sent events as `APIObject`s and releases the connection
    when exhausted, closed, or used as a context manager."""

    def __init__(self, response: http.client.HTTPResponse) -> None:
        self._response = response
        self._closed = False

    def __iter__(self) -> Iterator[APIObject]:
        data_lines: list[str] = []
        try:
            for raw in self._response:
                line = raw.decode("utf-8").rstrip("\r\n")
                if line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
                    continue
                if line or not data_lines:
                    continue
                data, data_lines = "\n".join(data_lines), []
                if data == "[DONE]":
                    return
                payload = json.loads(data)
                if isinstance(payload, dict) and "error" in payload and "choices" not in payload:
                    error = payload["error"] if isinstance(payload["error"], dict) else {}
                    raise APIStatusError(500, str(error.get("message", payload["error"])), payload)
                yield APIObject(payload)
        except (OSError, http.client.HTTPException) as error:
            raise APIConnectionError(f"the stream broke: {error}") from error
        finally:
            self.close()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._response.close()

    def __enter__(self) -> Stream:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _status_error(error: urllib.error.HTTPError) -> APIStatusError:
    raw = error.read().decode("utf-8", errors="replace")
    try:
        body: Any = json.loads(raw)
        message = body["error"]["message"]
    except (ValueError, KeyError, TypeError):
        body, message = raw, raw or error.reason
    return APIStatusError(error.code, str(message), body)


def _body(base: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(base)
    extra = params.get("extra_body") or {}
    body.update(
        {key: value for key, value in params.items() if value is not None and key != "extra_body"}
    )
    body.update(extra)
    return body


class Client:
    """OpenAI-shaped client: `chat.completions`, `completions` and `models`."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000/v1",
        api_key: str | None = None,
        timeout: float = 600.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.chat = _Chat(self)
        self.completions = _Completions(self)
        self.models = _Models(self)

    def _request(
        self, method: str, path: str, body: Mapping[str, Any] | None = None, *, stream: bool = False
    ) -> APIObject | Stream:
        headers = {"Accept": "text/event-stream" if stream else "application/json"}
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.base_url + path, data=data, headers=headers, method=method
        )
        try:
            response = urllib.request.urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as error:
            raise _status_error(error) from None
        except (urllib.error.URLError, OSError) as error:
            reason = getattr(error, "reason", error)
            raise APIConnectionError(f"cannot reach {self.base_url}: {reason}") from error
        if stream:
            return Stream(response)
        with response:
            return APIObject(json.loads(response.read()))


class _ChatCompletions:
    def __init__(self, client: Client) -> None:
        self._client = client

    def create(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        stream: bool = False,
        **params: Any,
    ) -> APIObject | Stream:
        """Create a chat completion. `params` takes OpenAI fields such as
        `max_tokens`, `temperature`, `top_p`, `seed`, `stop` and `stream_options`,
        server extensions such as `top_k` and `chat_template_kwargs`, and an
        `extra_body` dict merged into the request as is."""

        body = _body({"model": model, "messages": list(messages), "stream": stream}, params)
        return self._client._request("POST", "/chat/completions", body, stream=stream)


class _Chat:
    def __init__(self, client: Client) -> None:
        self.completions = _ChatCompletions(client)


class _Completions:
    def __init__(self, client: Client) -> None:
        self._client = client

    def create(
        self, *, model: str, prompt: str | Sequence[int], stream: bool = False, **params: Any
    ) -> APIObject | Stream:
        """Create a raw text completion from a string or a list of token IDs."""

        prompt_value = prompt if isinstance(prompt, str) else list(prompt)
        body = _body({"model": model, "prompt": prompt_value, "stream": stream}, params)
        return self._client._request("POST", "/completions", body, stream=stream)


class _Models:
    def __init__(self, client: Client) -> None:
        self._client = client

    def list(self) -> APIObject:
        return self._client._request("GET", "/models")

    def retrieve(self, model: str) -> APIObject:
        return self._client._request("GET", f"/models/{model}")
