from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Protocol, Sequence

import torch


@dataclass(frozen=True)
class GenerationConfig:
    """Sampling and stopping controls for one autoregressive response."""

    max_new_tokens: int = 128
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    stop_token_ids: tuple[int, ...] = ()
    seed: int | None = None
    max_context_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.seed is not None and self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.max_context_tokens is not None and self.max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be positive")
        if any(token_id < 0 for token_id in self.stop_token_ids):
            raise ValueError("stop_token_ids must be non-negative")


@dataclass
class ModelStep:
    """Logits and opaque backend state returned by prefill or decode."""

    logits: torch.Tensor
    state: Any


class AutoregressiveBackend(Protocol):
    """Minimal model-runner contract used by the generation loop."""

    def prefill(self, input_ids: Sequence[int]) -> ModelStep: ...

    def decode(self, token_id: int, state: Any) -> ModelStep: ...


def _last_token_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 1:
        result = logits
    elif logits.ndim == 2:
        if logits.shape[0] == 0:
            raise ValueError("model returned an empty logits sequence")
        result = logits[-1]
    elif logits.ndim == 3:
        if logits.shape[0] != 1:
            raise ValueError("generation currently supports batch size one")
        if logits.shape[1] == 0:
            raise ValueError("model returned an empty logits sequence")
        result = logits[0, -1]
    else:
        raise ValueError("logits must have shape [vocab], [tokens, vocab], or [1, tokens, vocab]")
    if result.numel() == 0:
        raise ValueError("model returned an empty vocabulary")
    return result


def sample_token(
    logits: torch.Tensor,
    config: GenerationConfig,
    *,
    generator: torch.Generator | None = None,
) -> int:
    """Select one token from last-token logits."""

    scores = _last_token_logits(logits)
    if config.temperature == 0:
        return int(torch.argmax(scores).item())

    scores = scores.float() / config.temperature
    if config.top_k:
        top_k = min(config.top_k, scores.numel())
        top_values, top_indices = torch.topk(scores, top_k)
        scores = torch.full_like(scores, float("-inf")).scatter(0, top_indices, top_values)

    probabilities = torch.softmax(scores, dim=-1)
    if config.top_p < 1.0:
        sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)
        cumulative = torch.cumsum(sorted_probabilities, dim=-1)
        remove = cumulative - sorted_probabilities >= config.top_p
        sorted_probabilities = sorted_probabilities.masked_fill(remove, 0.0)
        probabilities = torch.zeros_like(probabilities).scatter(
            0, sorted_indices, sorted_probabilities
        )
        probabilities = probabilities / probabilities.sum()

    return int(torch.multinomial(probabilities, 1, generator=generator).item())


def generate_tokens(
    backend: AutoregressiveBackend,
    prompt_ids: Sequence[int],
    config: GenerationConfig | None = None,
) -> Iterator[int]:
    """Yield generated token IDs, including a terminal stop token when sampled."""

    options = config or GenerationConfig()
    prompt = tuple(int(token_id) for token_id in prompt_ids)
    if not prompt:
        raise ValueError("prompt_ids must contain at least one token")
    if any(token_id < 0 for token_id in prompt):
        raise ValueError("prompt_ids must be non-negative")

    backend_context_tokens = getattr(backend, "max_context_tokens", None)
    if backend_context_tokens is not None and (
        not isinstance(backend_context_tokens, int) or backend_context_tokens <= 0
    ):
        raise ValueError("backend max_context_tokens must be a positive integer")
    context_limits = [
        limit for limit in (options.max_context_tokens, backend_context_tokens) if limit is not None
    ]
    effective_context_tokens = min(context_limits) if context_limits else None
    if effective_context_tokens is not None and len(prompt) > effective_context_tokens:
        raise ValueError("prompt exceeds max_context_tokens")
    if options.max_new_tokens == 0:
        return
    if effective_context_tokens is not None and len(prompt) == effective_context_tokens:
        return

    step = backend.prefill(prompt)
    random_generator: torch.Generator | None = None
    if options.temperature > 0 and options.seed is not None:
        random_generator = torch.Generator(device=step.logits.device)
        random_generator.manual_seed(options.seed)

    stop_ids = set(options.stop_token_ids)
    for token_index in range(options.max_new_tokens):
        if (
            effective_context_tokens is not None
            and len(prompt) + token_index >= effective_context_tokens
        ):
            return
        token_id = sample_token(step.logits, options, generator=random_generator)
        yield token_id
        context_is_full = (
            effective_context_tokens is not None
            and len(prompt) + token_index + 1 >= effective_context_tokens
        )
        if token_id in stop_ids or token_index + 1 == options.max_new_tokens or context_is_full:
            return
        step = backend.decode(token_id, step.state)


@dataclass(frozen=True)
class EngineGenerationState:
    generation: int
    position: int
    _owner: object


class EngineCausalLM:
    """Single-session adapter for fixed one-token DLI causal-LM graphs.

    Prompt tokens are intentionally submitted one at a time. If a distinct
    prefill graph is provided, it runs the first prompt token and the decode
    graph runs all remaining prompt and generated tokens. The engine cache is
    reset at prefill and retained across subsequent decode calls.
    """

    def __init__(
        self,
        engine: Any,
        graph: Any,
        static_inputs: Mapping[str, torch.Tensor],
        *,
        input_name: str = "input_ids",
        output_name: str = "logits",
        device: torch.device | str | None = None,
        max_context_tokens: int | None = None,
        state: Any | None = None,
        prefill_graph: Any | None = None,
        prefill_static_inputs: Mapping[str, torch.Tensor] | None = None,
        input_shape: Sequence[int] = (1,),
    ) -> None:
        if max_context_tokens is not None and max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be positive")
        normalized_input_shape = tuple(input_shape)
        if not normalized_input_shape or any(
            isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0
            for dimension in normalized_input_shape
        ):
            raise ValueError("input_shape must contain positive integer dimensions")
        input_elements = 1
        for dimension in normalized_input_shape:
            input_elements *= dimension
        if input_elements != 1:
            raise ValueError("input_shape must describe exactly one token")
        self.engine = engine
        self.graph = graph
        self.static_inputs = dict(static_inputs)
        self.prefill_graph = graph if prefill_graph is None else prefill_graph
        self.prefill_static_inputs = (
            self.static_inputs if prefill_static_inputs is None else dict(prefill_static_inputs)
        )
        self.input_name = input_name
        self.input_shape = normalized_input_shape
        self.output_name = output_name
        self.max_context_tokens = max_context_tokens
        self.state = state
        self.device = torch.device(device) if device is not None else self._input_device()
        self._state_owner = object()
        self._generation = 0
        self._position = 0

    def _input_device(self) -> torch.device:
        for inputs in (self.static_inputs, self.prefill_static_inputs):
            for tensor in inputs.values():
                if isinstance(tensor, torch.Tensor):
                    return tensor.device
        raise ValueError("device is required when static_inputs contains no tensors")

    def reset(self) -> None:
        if self.state is None:
            self.engine.reset()
        else:
            self.state.reset()
        self._generation += 1
        self._position = 0

    def _run_token(
        self,
        token_id: int,
        graph: Any,
        static_inputs: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        if token_id < 0:
            raise ValueError("token_id must be non-negative")
        if self.max_context_tokens is not None and self._position >= self.max_context_tokens:
            raise ValueError("model context is full")
        inputs = dict(static_inputs)
        inputs[self.input_name] = torch.full(
            self.input_shape,
            token_id,
            dtype=torch.int64,
            device=self.device,
        )
        run_options: dict[str, Any] = {"position_offset": self._position}
        if self.state is not None:
            run_options["state"] = self.state
        try:
            outputs = self.engine.run(graph, inputs, **run_options)
            if self.output_name not in outputs:
                raise KeyError(f"graph did not return logits output: {self.output_name}")
            logits = outputs[self.output_name]
        except Exception:
            # A graph may update some layers before a later operator fails.
            # Clear the request state so callers cannot resume a partial step.
            self.reset()
            raise
        self._position += 1
        return logits

    def prefill(self, input_ids: Sequence[int]) -> ModelStep:
        prompt = tuple(int(token_id) for token_id in input_ids)
        if not prompt:
            raise ValueError("input_ids must contain at least one token")
        if self.max_context_tokens is not None and len(prompt) > self.max_context_tokens:
            raise ValueError("prompt exceeds model context")
        self.reset()
        logits = self._run_token(
            prompt[0],
            self.prefill_graph,
            self.prefill_static_inputs,
        )
        for token_id in prompt[1:]:
            logits = self._run_token(token_id, self.graph, self.static_inputs)
        return ModelStep(
            logits,
            EngineGenerationState(self._generation, self._position, self._state_owner),
        )

    def decode(self, token_id: int, state: Any) -> ModelStep:
        if not isinstance(state, EngineGenerationState):
            raise TypeError("decode state was not created by EngineCausalLM")
        if (
            state._owner is not self._state_owner
            or state.generation != self._generation
            or state.position != self._position
        ):
            raise ValueError("decode state is stale or belongs to another generation")
        logits = self._run_token(int(token_id), self.graph, self.static_inputs)
        return ModelStep(
            logits,
            EngineGenerationState(self._generation, self._position, self._state_owner),
        )


__all__ = [
    "AutoregressiveBackend",
    "EngineCausalLM",
    "EngineGenerationState",
    "GenerationConfig",
    "ModelStep",
    "generate_tokens",
    "sample_token",
]
