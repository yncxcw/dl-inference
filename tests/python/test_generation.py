from __future__ import annotations

from typing import Any, Sequence

import torch

from dli.generation import (
    EngineCausalLM,
    GenerationConfig,
    ModelStep,
    generate_tokens,
    sample_token,
)


class FakeBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    @staticmethod
    def _logits(token_id: int) -> torch.Tensor:
        logits = torch.full((5,), -10.0)
        logits[token_id] = 10.0
        return logits

    def prefill(self, input_ids: Sequence[int]) -> ModelStep:
        self.calls.append(("prefill", tuple(input_ids)))
        return ModelStep(self._logits(2), "after-prefill")

    def decode(self, token_id: int, state: Any) -> ModelStep:
        self.calls.append(("decode", (token_id, state)))
        if token_id == 2:
            return ModelStep(self._logits(3), "after-two")
        return ModelStep(self._logits(1), "after-three")


class FakeEngine:
    def __init__(self) -> None:
        self.reset_count = 0
        self.calls: list[tuple[int, int]] = []

    def reset(self) -> None:
        self.reset_count += 1

    def run(
        self,
        graph: object,
        inputs: dict[str, torch.Tensor],
        *,
        state: object | None = None,
        position_offset: int = 0,
    ):
        del graph
        del state
        token_id = int(inputs["input_ids"].item())
        self.calls.append((token_id, position_offset))
        logits = torch.zeros(6)
        logits[(token_id + 1) % logits.numel()] = 1.0
        return {"logits": logits}


class FakeState:
    def __init__(self) -> None:
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1


class FailingEngine(FakeEngine):
    def run(self, graph: object, inputs: dict[str, torch.Tensor], **kwargs):
        del graph, inputs, kwargs
        raise RuntimeError("operator failed after a state update")


class MissingOutputEngine(FakeEngine):
    def __init__(self) -> None:
        super().__init__()
        self.omit_output = False

    def run(self, graph: object, inputs: dict[str, torch.Tensor], **kwargs):
        outputs = super().run(graph, inputs, **kwargs)
        return {} if self.omit_output else outputs


def test_greedy_generation_stops_on_eos() -> None:
    backend = FakeBackend()
    tokens = list(
        generate_tokens(
            backend,
            [4, 0],
            GenerationConfig(max_new_tokens=10, stop_token_ids=(1,)),
        )
    )
    assert tokens == [2, 3, 1]
    assert backend.calls == [
        ("prefill", (4, 0)),
        ("decode", (2, "after-prefill")),
        ("decode", (3, "after-two")),
    ]


def test_generation_limits_and_zero_token_shortcut() -> None:
    backend = FakeBackend()
    assert list(generate_tokens(backend, [0], GenerationConfig(max_new_tokens=2))) == [2, 3]
    assert len(backend.calls) == 2

    unused = FakeBackend()
    assert list(generate_tokens(unused, [0], GenerationConfig(max_new_tokens=0))) == []
    assert unused.calls == []

    capped = FakeBackend()
    assert list(
        generate_tokens(
            capped,
            [0, 1],
            GenerationConfig(max_new_tokens=5, max_context_tokens=3),
        )
    ) == [2]

    backend_capped = FakeBackend()
    backend_capped.max_context_tokens = 3
    assert list(generate_tokens(backend_capped, [0, 1], GenerationConfig(max_new_tokens=5))) == [2]
    assert backend_capped.calls == [("prefill", (0, 1))]

    full_backend = FakeBackend()
    full_backend.max_context_tokens = 2
    assert list(generate_tokens(full_backend, [0, 1])) == []
    assert full_backend.calls == []


def test_sampling_filters_and_validation() -> None:
    logits = torch.tensor([1.0, 2.0, 3.0])
    assert sample_token(logits, GenerationConfig(temperature=1.0, top_k=1)) == 2
    assert sample_token(logits.reshape(1, 1, 3), GenerationConfig()) == 2

    for kwargs in (
        {"max_new_tokens": -1},
        {"temperature": -1.0},
        {"top_k": -1},
        {"top_p": 0.0},
        {"top_p": 1.1},
        {"max_context_tokens": 0},
    ):
        try:
            GenerationConfig(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid config was accepted: {kwargs}")


def test_seeded_sampling_is_repeatable() -> None:
    first = list(
        generate_tokens(
            FakeBackend(),
            [0],
            GenerationConfig(max_new_tokens=3, temperature=2.0, seed=1234),
        )
    )
    second = list(
        generate_tokens(
            FakeBackend(),
            [0],
            GenerationConfig(max_new_tokens=3, temperature=2.0, seed=1234),
        )
    )
    assert first == second


def test_engine_backend_tracks_positions_and_rejects_stale_state() -> None:
    engine = FakeEngine()
    state = FakeState()
    backend = EngineCausalLM(
        engine,
        object(),
        {"weight": torch.zeros(1)},
        max_context_tokens=5,
        state=state,
    )
    step = backend.prefill([2, 4])
    assert state.reset_count == 1
    assert engine.reset_count == 0
    assert engine.calls == [(2, 0), (4, 1)]
    next_step = backend.decode(5, step.state)
    assert engine.calls[-1] == (5, 2)

    try:
        backend.decode(1, step.state)
    except ValueError as error:
        assert "stale" in str(error)
    else:
        raise AssertionError("stale state was accepted")

    backend.prefill([1])
    assert state.reset_count == 2
    try:
        backend.decode(1, next_step.state)
    except ValueError as error:
        assert "stale" in str(error)
    else:
        raise AssertionError("state from a previous prefill was accepted")


def test_engine_backend_rejects_state_from_another_adapter() -> None:
    first = EngineCausalLM(FakeEngine(), object(), {"weight": torch.zeros(1)})
    second_engine = FakeEngine()
    second = EngineCausalLM(second_engine, object(), {"weight": torch.zeros(1)})
    foreign_state = first.prefill([1]).state
    second.prefill([2])

    try:
        second.decode(3, foreign_state)
    except ValueError as error:
        assert "another generation" in str(error)
    else:
        raise AssertionError("state from another EngineCausalLM adapter was accepted")
    assert second_engine.calls == [(2, 0)]


def test_prompt_validation() -> None:
    try:
        list(generate_tokens(FakeBackend(), []))
    except ValueError as error:
        assert "prompt_ids" in str(error)
    else:
        raise AssertionError("empty prompt was accepted")


def test_engine_backend_clears_partial_state_after_failure() -> None:
    state = FakeState()
    backend = EngineCausalLM(
        FailingEngine(),
        object(),
        {"weight": torch.zeros(1)},
        state=state,
    )
    try:
        backend.prefill([1])
    except RuntimeError as error:
        assert "operator failed" in str(error)
    else:
        raise AssertionError("engine failure was not propagated")
    assert state.reset_count == 2


def test_engine_backend_clears_state_when_logits_output_is_missing() -> None:
    engine = MissingOutputEngine()
    state = FakeState()
    backend = EngineCausalLM(
        engine,
        object(),
        {"weight": torch.zeros(1)},
        state=state,
    )
    step = backend.prefill([1])
    engine.omit_output = True

    try:
        backend.decode(2, step.state)
    except KeyError as error:
        assert "logits" in str(error)
    else:
        raise AssertionError("missing logits output was accepted")

    assert state.reset_count == 2
    try:
        backend.decode(2, step.state)
    except ValueError as error:
        assert "stale" in str(error)
    else:
        raise AssertionError("state from a failed decode was accepted")


if __name__ == "__main__":
    test_greedy_generation_stops_on_eos()
    test_generation_limits_and_zero_token_shortcut()
    test_sampling_filters_and_validation()
    test_seeded_sampling_is_repeatable()
    test_engine_backend_tracks_positions_and_rejects_stale_state()
    test_engine_backend_rejects_state_from_another_adapter()
    test_prompt_validation()
    test_engine_backend_clears_partial_state_after_failure()
    test_engine_backend_clears_state_when_logits_output_is_missing()
