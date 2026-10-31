"""Iteration-level scheduler that interleaves batch-1 model steps across requests."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence, Union

import torch

from dli.generation import GenerationConfig, sample_token

from .kv_cache import PrefixCache
from .text import IncrementalDecoder, StopStringFilter

logger = logging.getLogger(__name__)


class ModelRunner(Protocol):
    """One-token model step over an opaque, copyable request state."""

    max_context_tokens: int | None

    def new_state(self) -> Any: ...

    def step(self, state: Any, token_id: int, position: int) -> torch.Tensor:
        """Consume `token_id` at `position`, advance `state`, return next-token logits."""
        ...


@dataclass(frozen=True)
class SamplingParams:
    max_tokens: int
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    seed: int | None = None
    stop: tuple[str, ...] = ()
    stop_token_ids: tuple[int, ...] = ()

    def generation_config(self) -> GenerationConfig:
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        return GenerationConfig(
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            stop_token_ids=self.stop_token_ids,
            seed=self.seed,
        )


@dataclass(frozen=True)
class GenerationRequest:
    prompt_ids: tuple[int, ...]
    params: SamplingParams
    # Prompt lengths worth snapshotting besides the full prompt, such as the
    # end of the chat history before the generation prompt.
    snapshot_points: tuple[int, ...] = ()


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class Finished:
    finish_reason: str  # "stop", "length" or "abort"
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int


@dataclass(frozen=True)
class Failed:
    error: BaseException


Event = Union[TextDelta, Finished, Failed]


class RequestHandle:
    """Lets the submitter cancel a request; it takes effect at the next step."""

    def __init__(self) -> None:
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        self._cancelled.set()

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()


@dataclass
class _Sequence:
    request: GenerationRequest
    handle: RequestHandle
    emit: Callable[[Event], None]
    config: GenerationConfig
    decoder: IncrementalDecoder
    stops: StopStringFilter
    state: Any = None
    position: int = 0  # tokens consumed by `state`
    cached_tokens: int = 0
    logits: Any = None  # next-token logits after the last consumed token
    phase: str = "prefill"  # "prefill" -> "sample" <-> "decode"
    pending_token: int | None = None  # sampled but not yet consumed
    generated: list[int] = field(default_factory=list)
    generator: torch.Generator | None = None


class Scheduler:
    """Owns the model and runs every step on one worker thread.

    Each round admits waiting requests up to `max_num_seqs`, then gives every
    active sequence one step: its next prompt token while prefilling,
    otherwise its last sampled token. Batch-1 graphs cannot batch requests,
    but interleaving lets every client stream instead of queueing behind
    another reply. `step_round` also lets tests drive the loop synchronously.
    """

    def __init__(
        self,
        runner: ModelRunner,
        decode: Callable[[Sequence[int]], str],
        *,
        max_num_seqs: int = 4,
        prefix_cache_size: int = 8,
    ) -> None:
        if max_num_seqs <= 0:
            raise ValueError("max_num_seqs must be positive")
        self.runner = runner
        self.max_num_seqs = max_num_seqs
        self.prefix_cache = PrefixCache(prefix_cache_size)
        self._decode = decode
        self._pending: deque[_Sequence] = deque()
        self._active: list[_Sequence] = []
        self._condition = threading.Condition()
        self._thread: threading.Thread | None = None
        self._stopping = False
        self._counters = dict.fromkeys(
            (
                "requests",
                "finished",
                "aborted",
                "failed",
                "prompt_tokens",
                "cached_tokens",
                "generated_tokens",
                "model_steps",
            ),
            0,
        )
        self._step_seconds = 0.0

    # Submission side, called from any thread.

    def submit(self, request: GenerationRequest, emit: Callable[[Event], None]) -> RequestHandle:
        """Queue a request; `emit` receives its events on the worker thread."""

        prompt = request.prompt_ids
        if not prompt:
            raise ValueError("the prompt must contain at least one token")
        capacity = self.runner.max_context_tokens
        if capacity is not None and len(prompt) > capacity:
            raise ValueError(
                f"the prompt has {len(prompt)} tokens, more than the model's context "
                f"window of {capacity} tokens"
            )
        sequence = _Sequence(
            request=request,
            handle=RequestHandle(),
            emit=emit,
            config=request.params.generation_config(),
            decoder=IncrementalDecoder(self._decode),
            stops=StopStringFilter(request.params.stop),
        )
        with self._condition:
            if self._stopping:
                raise RuntimeError("the server is shutting down")
            self._pending.append(sequence)
            self._counters["requests"] += 1
            self._condition.notify()
        return sequence.handle

    def stats(self) -> dict[str, Any]:
        with self._condition:
            pending = len(self._pending)
        steps = self._counters["model_steps"]
        return {
            **self._counters,
            "active": len(self._active),
            "pending": pending,
            "max_num_seqs": self.max_num_seqs,
            "max_context_tokens": self.runner.max_context_tokens,
            "mean_step_ms": round(1000.0 * self._step_seconds / steps, 3) if steps else None,
            "prefix_cache": self.prefix_cache.stats(),
        }

    # Worker lifecycle.

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("the scheduler is already running")
        self._thread = threading.Thread(target=self._run, name="dli-scheduler", daemon=True)
        self._thread.start()

    def shutdown(self, timeout: float | None = 10.0) -> None:
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None
        else:
            self._fail_outstanding()

    def has_work(self) -> bool:
        with self._condition:
            return bool(self._pending or self._active)

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._stopping and not self._pending and not self._active:
                    self._condition.wait()
                if self._stopping:
                    break
            self.step_round()
        self._fail_outstanding()

    def _fail_outstanding(self) -> None:
        with self._condition:
            outstanding = self._active + list(self._pending)
            self._active.clear()
            self._pending.clear()
        for sequence in outstanding:
            self._emit(sequence, Failed(RuntimeError("the server is shutting down")))

    # Worker side.

    def step_round(self) -> None:
        self._admit()
        for sequence in list(self._active):
            self._step(sequence)

    def _admit(self) -> None:
        while len(self._active) < self.max_num_seqs:
            with self._condition:
                if not self._pending:
                    return
                sequence = self._pending.popleft()
            prompt = sequence.request.prompt_ids
            if sequence.handle.cancelled:
                self._counters["aborted"] += 1
                self._emit(sequence, Finished("abort", len(prompt), 0, 0))
                continue
            match = self.prefix_cache.lookup(prompt)
            if match is None:
                sequence.state = self.runner.new_state()
            else:
                sequence.state = match.state
                sequence.position = sequence.cached_tokens = match.tokens
                if match.tokens == len(prompt):
                    sequence.logits = match.logits
                    sequence.phase = "sample"
            self._counters["prompt_tokens"] += len(prompt)
            self._counters["cached_tokens"] += sequence.cached_tokens
            self._active.append(sequence)

    def _step(self, sequence: _Sequence) -> None:
        if sequence.handle.cancelled:
            self._finish(sequence, "abort")
            return
        try:
            if sequence.phase == "prefill":
                prompt = sequence.request.prompt_ids
                self._consume(sequence, prompt[sequence.position])
                if sequence.position < len(prompt):
                    if sequence.position in sequence.request.snapshot_points:
                        self.prefix_cache.insert(
                            prompt[: sequence.position], sequence.state, sequence.logits
                        )
                    return
                self.prefix_cache.insert(prompt, sequence.state, sequence.logits)
            elif sequence.phase == "decode":
                self._consume(sequence, sequence.pending_token)
                sequence.pending_token = None
            self._sample(sequence)
        except Exception as error:
            logger.exception("request failed during a model step")
            self._remove(sequence)
            self._counters["failed"] += 1
            self._emit(sequence, Failed(error))

    def _consume(self, sequence: _Sequence, token_id: int) -> None:
        started = time.perf_counter()
        sequence.logits = self.runner.step(sequence.state, token_id, sequence.position)
        self._step_seconds += time.perf_counter() - started
        self._counters["model_steps"] += 1
        sequence.position += 1

    def _sample(self, sequence: _Sequence) -> None:
        config = sequence.config
        if sequence.generator is None and config.temperature > 0 and config.seed is not None:
            sequence.generator = torch.Generator(device=sequence.logits.device)
            sequence.generator.manual_seed(config.seed)
        token_id = sample_token(sequence.logits, config, generator=sequence.generator)
        sequence.generated.append(token_id)
        self._counters["generated_tokens"] += 1
        if token_id in config.stop_token_ids:
            self._finish(sequence, "stop")
            return
        text, stopped = sequence.stops.feed(sequence.decoder.push(token_id))
        if text:
            self._emit(sequence, TextDelta(text))
        if stopped:
            self._finish(sequence, "stop", flush=False)
            return
        capacity = self.runner.max_context_tokens
        if len(sequence.generated) >= config.max_new_tokens or (
            capacity is not None and sequence.position >= capacity
        ):
            self._finish(sequence, "length")
            return
        sequence.pending_token = token_id
        sequence.phase = "decode"

    def _finish(self, sequence: _Sequence, reason: str, *, flush: bool = True) -> None:
        prompt = sequence.request.prompt_ids
        if reason != "abort":
            if flush:
                text, stopped = sequence.stops.feed(sequence.decoder.flush())
                tail = text if stopped else text + sequence.stops.flush()
                if tail:
                    self._emit(sequence, TextDelta(tail))
                if stopped:
                    reason = "stop"
            # Snapshot everything consumed, so a follow-up prompt that repeats
            # this reply verbatim resumes after it.
            consumed = sequence.position - len(prompt)
            if consumed > 0:
                tokens = prompt + tuple(sequence.generated[:consumed])
                self.prefix_cache.insert(tokens, sequence.state, sequence.logits)
        self._remove(sequence)
        self._counters["aborted" if reason == "abort" else "finished"] += 1
        self._emit(
            sequence,
            Finished(reason, len(prompt), len(sequence.generated), sequence.cached_tokens),
        )

    def _remove(self, sequence: _Sequence) -> None:
        if sequence in self._active:
            self._active.remove(sequence)

    def _emit(self, sequence: _Sequence, event: Event) -> None:
        try:
            sequence.emit(event)
        except Exception:
            # The consumer is gone (for example its event loop closed); stop
            # spending model steps on it.
            logger.exception("dropping a request whose consumer failed")
            sequence.handle.cancel()
