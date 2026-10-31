"""OpenAI-compatible HTTP API over the DLI scheduler (FastAPI)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Literal, Mapping, Protocol, Sequence

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .scheduler import (
    Event,
    Failed,
    Finished,
    GenerationRequest,
    SamplingParams,
    Scheduler,
    TextDelta,
)

STATIC_DIR = Path(__file__).with_name("static")


class Tokenizer(Protocol):
    stop_token_ids: tuple[int, ...]

    def encode(self, text: str) -> list[int]: ...

    def encode_chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        add_generation_prompt: bool,
        template_kwargs: Mapping[str, Any],
    ) -> list[int]: ...

    def decode(self, token_ids: Sequence[int]) -> str: ...


@dataclass(frozen=True)
class SamplingDefaults:
    """Values used when a request omits them (vLLM reads them from the model)."""

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0


class APIError(Exception):
    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        type: str = "invalid_request_error",
        param: str | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = {"error": {"message": message, "type": type, "param": param, "code": code}}


# Request schemas. Unknown OpenAI fields are accepted and ignored.


class StreamOptions(BaseModel):
    include_usage: bool = False


class ContentPart(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: str
    text: str | None = None


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: Literal["system", "developer", "user", "assistant"]
    content: str | list[ContentPart] | None = None


class _GenerationBody(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: str
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=0)
    seed: int | None = Field(default=None, ge=0)
    stop: str | list[str] | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    n: int = 1
    logprobs: bool | int | None = None


class ChatCompletionBody(_GenerationBody):
    messages: list[ChatMessage] = Field(min_length=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    tools: list[Any] | None = None
    chat_template_kwargs: dict[str, Any] | None = None


class CompletionBody(_GenerationBody):
    prompt: str | list[int]
    echo: bool = False


def _sse(payload: Mapping[str, Any] | str) -> str:
    data = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return f"data: {data}\n\n"


def _usage(finished: Finished) -> dict[str, Any]:
    return {
        "prompt_tokens": finished.prompt_tokens,
        "completion_tokens": finished.completion_tokens,
        "total_tokens": finished.prompt_tokens + finished.completion_tokens,
        "prompt_tokens_details": {"cached_tokens": finished.cached_tokens},
    }


def _message_text(message: ChatMessage) -> str:
    if message.content is None:
        return ""
    if isinstance(message.content, str):
        return message.content
    parts = []
    for part in message.content:
        if part.type != "text" or part.text is None:
            raise APIError(400, f"content parts of type {part.type!r} are not supported")
        parts.append(part.text)
    return "".join(parts)


def create_app(
    scheduler: Scheduler,
    tokenizer: Tokenizer,
    *,
    model_name: str,
    api_key: str | None = None,
    allow_origins: Sequence[str] = (),
    defaults: SamplingDefaults = SamplingDefaults(),
    default_max_tokens: int | None = None,
    enable_thinking: bool = False,
    manage_scheduler: bool = True,
) -> FastAPI:
    """Build the app. With `manage_scheduler`, its lifespan starts and stops the worker."""

    created = int(time.time())
    capacity = scheduler.runner.max_context_tokens

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if manage_scheduler:
            scheduler.start()
        try:
            yield
        finally:
            if manage_scheduler:
                scheduler.shutdown()

    app = FastAPI(title="DLI OpenAI-compatible server", lifespan=lifespan)
    if allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(allow_origins),
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.exception_handler(APIError)
    async def _api_error(_: Request, error: APIError) -> JSONResponse:
        return JSONResponse(error.body, status_code=error.status_code)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, error: RequestValidationError) -> JSONResponse:
        problem = error.errors()[0] if error.errors() else {}
        location = ".".join(str(part) for part in problem.get("loc", ())[1:])
        message = problem.get("msg", "invalid request")
        return JSONResponse(
            APIError(400, f"{location}: {message}" if location else message, param=location).body,
            status_code=400,
        )

    def require_api_key(request: Request) -> None:
        if api_key is not None and request.headers.get("authorization") != f"Bearer {api_key}":
            raise APIError(401, "invalid or missing API key", type="authentication_error")

    def model_card() -> dict[str, Any]:
        return {
            "id": model_name,
            "object": "model",
            "created": created,
            "owned_by": "dli",
            "max_model_len": capacity,
        }

    def check_body(body: _GenerationBody) -> None:
        if body.model != model_name:
            raise APIError(
                404,
                f"the model {body.model!r} does not exist",
                param="model",
                code="model_not_found",
            )
        if body.n != 1:
            raise APIError(400, "only n=1 is supported", param="n")
        if body.logprobs:
            raise APIError(400, "logprobs are not supported", param="logprobs")

    def sampling(
        body: _GenerationBody, prompt: Sequence[int], max_tokens: int | None
    ) -> SamplingParams:
        if capacity is not None and len(prompt) >= capacity:
            raise APIError(
                400,
                f"this model's maximum context length is {capacity} tokens, and the prompt "
                f"has {len(prompt)} tokens, leaving no room for a reply",
                code="context_length_exceeded",
            )
        room = None if capacity is None else capacity - len(prompt)
        limits = [value for value in (max_tokens, default_max_tokens, room) if value is not None]
        stop = (body.stop,) if isinstance(body.stop, str) else tuple(body.stop or ())
        return SamplingParams(
            max_tokens=min(limits) if limits else 1024,
            temperature=defaults.temperature if body.temperature is None else body.temperature,
            top_p=defaults.top_p if body.top_p is None else body.top_p,
            top_k=defaults.top_k if body.top_k is None else body.top_k,
            seed=body.seed,
            stop=stop,
            stop_token_ids=tokenizer.stop_token_ids,
        )

    def submit(request: GenerationRequest) -> tuple[Any, asyncio.Queue[Event]]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Event] = asyncio.Queue()
        try:
            handle = scheduler.submit(
                request, lambda event: loop.call_soon_threadsafe(queue.put_nowait, event)
            )
        except ValueError as error:
            raise APIError(400, str(error)) from error
        except RuntimeError as error:
            raise APIError(503, str(error), type="server_error") from error
        return handle, queue

    async def events(handle: Any, queue: asyncio.Queue[Event]) -> AsyncIterator[Event]:
        try:
            while True:
                event = await queue.get()
                yield event
                if not isinstance(event, TextDelta):
                    return
        finally:
            # Covers client disconnects; a no-op once the request finished.
            handle.cancel()

    async def collect(handle: Any, queue: asyncio.Queue[Event]) -> tuple[str, Finished]:
        text = []
        async for event in events(handle, queue):
            if isinstance(event, TextDelta):
                text.append(event.text)
            elif isinstance(event, Failed):
                raise APIError(500, f"generation failed: {event.error}", type="server_error")
            else:
                return "".join(text), event
        raise APIError(500, "generation ended without a result", type="server_error")

    def stream(handle: Any, queue: asyncio.Queue[Event], chunk, include_usage: bool, first=None):
        async def body() -> AsyncIterator[str]:
            if first is not None:
                yield _sse(first)
            async for event in events(handle, queue):
                if isinstance(event, TextDelta):
                    yield _sse(chunk(text=event.text))
                elif isinstance(event, Failed):
                    yield _sse(
                        APIError(500, f"generation failed: {event.error}", type="server_error").body
                    )
                else:
                    yield _sse(chunk(finish_reason=event.finish_reason))
                    if include_usage:
                        yield _sse(chunk(usage=_usage(event)))
            yield _sse("[DONE]")

        return StreamingResponse(
            body(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/stats")
    async def stats() -> dict[str, Any]:
        return {"model": model_name, **scheduler.stats()}

    @app.get("/v1/models", dependencies=[Depends(require_api_key)])
    async def list_models() -> dict[str, Any]:
        return {"object": "list", "data": [model_card()]}

    @app.get("/v1/models/{model_id:path}", dependencies=[Depends(require_api_key)])
    async def get_model(model_id: str) -> dict[str, Any]:
        if model_id != model_name:
            raise APIError(404, f"the model {model_id!r} does not exist", code="model_not_found")
        return model_card()

    @app.post("/v1/chat/completions", dependencies=[Depends(require_api_key)])
    async def chat_completions(body: ChatCompletionBody):
        check_body(body)
        if body.tools:
            raise APIError(400, "tools are not supported", param="tools")
        messages = [
            {"role": "system" if m.role == "developer" else m.role, "content": _message_text(m)}
            for m in body.messages
        ]
        template_kwargs = {"enable_thinking": enable_thinking, **(body.chat_template_kwargs or {})}
        prompt = await asyncio.to_thread(
            tokenizer.encode_chat,
            messages,
            add_generation_prompt=True,
            template_kwargs=template_kwargs,
        )
        # Snapshot the end of the history as well: templates re-render earlier
        # replies, so the next turn's prompt reliably extends this point.
        snapshot_points: tuple[int, ...] = ()
        if messages[-1]["role"] != "assistant":
            history = await asyncio.to_thread(
                tokenizer.encode_chat,
                messages,
                add_generation_prompt=False,
                template_kwargs=template_kwargs,
            )
            if 0 < len(history) < len(prompt) and prompt[: len(history)] == history:
                snapshot_points = (len(history),)
        params = sampling(body, prompt, body.max_completion_tokens or body.max_tokens)
        handle, queue = submit(GenerationRequest(tuple(prompt), params, snapshot_points))
        response_id = f"chatcmpl-{uuid.uuid4().hex}"
        now = int(time.time())

        if not body.stream:
            text, finished = await collect(handle, queue)
            return {
                "id": response_id,
                "object": "chat.completion",
                "created": now,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": finished.finish_reason,
                        "logprobs": None,
                    }
                ],
                "usage": _usage(finished),
            }

        def chunk(*, text=None, finish_reason=None, usage=None, role=None):
            payload = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": now,
                "model": model_name,
                "choices": [],
            }
            if usage is not None:
                payload["usage"] = usage
                return payload
            delta: dict[str, str] = {"role": role, "content": ""} if role is not None else {}
            if text is not None:
                delta["content"] = text
            payload["choices"] = [
                {"index": 0, "delta": delta, "finish_reason": finish_reason, "logprobs": None}
            ]
            return payload

        include_usage = bool(body.stream_options and body.stream_options.include_usage)
        return stream(handle, queue, chunk, include_usage, first=chunk(role="assistant"))

    @app.post("/v1/completions", dependencies=[Depends(require_api_key)])
    async def completions(body: CompletionBody):
        check_body(body)
        if isinstance(body.prompt, str):
            prompt = await asyncio.to_thread(tokenizer.encode, body.prompt)
        else:
            prompt = list(body.prompt)
        if not prompt:
            raise APIError(400, "the prompt must not be empty", param="prompt")
        params = sampling(body, prompt, body.max_tokens)
        handle, queue = submit(GenerationRequest(tuple(prompt), params))
        response_id = f"cmpl-{uuid.uuid4().hex}"
        now = int(time.time())
        echoed = body.prompt if body.echo and isinstance(body.prompt, str) else ""

        if not body.stream:
            text, finished = await collect(handle, queue)
            return {
                "id": response_id,
                "object": "text_completion",
                "created": now,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "text": echoed + text,
                        "finish_reason": finished.finish_reason,
                        "logprobs": None,
                    }
                ],
                "usage": _usage(finished),
            }

        def chunk(*, text=None, finish_reason=None, usage=None):
            payload = {
                "id": response_id,
                "object": "text_completion",
                "created": now,
                "model": model_name,
                "choices": [],
            }
            if usage is not None:
                payload["usage"] = usage
                return payload
            payload["choices"] = [
                {"index": 0, "text": text or "", "finish_reason": finish_reason, "logprobs": None}
            ]
            return payload

        include_usage = bool(body.stream_options and body.stream_options.include_usage)
        first = chunk(text=echoed) if echoed else None
        return stream(handle, queue, chunk, include_usage, first=first)

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

        @app.get("/", include_in_schema=False)
        async def index() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")

    return app
