"""Deterministic stand-ins for the model runner and tokenizer in server tests."""

from __future__ import annotations

import threading
import time
from typing import Any, Mapping, Sequence

import torch

EOS = 3
ASSISTANT = 2  # generation-prompt marker that starts an assistant reply
VOCAB = 256


class FakeState:
    """Consumed tokens; `copy` returns an independent snapshot like ExecutionState."""

    def __init__(self, tokens: Sequence[int] = ()) -> None:
        self.tokens = list(tokens)

    def copy(self) -> FakeState:
        return FakeState(self.tokens)


class FakeRunner:
    """Replies with `reply` after the last assistant marker, then emits EOS.

    Logits depend only on the tokens a state has consumed, so prefix-cache
    snapshots behave like real request state.
    """

    def __init__(
        self,
        reply: str = "Hello there!",
        *,
        max_context_tokens: int | None = 512,
        step_delay: float = 0.0,
    ) -> None:
        self.reply = [ord(character) for character in reply]
        self.max_context_tokens = max_context_tokens
        self.step_delay = step_delay
        self.fail_on_token: int | None = None
        self.steps: list[tuple[int, int]] = []  # (id of the stepped state, token)
        self._lock = threading.Lock()

    def new_state(self) -> FakeState:
        return FakeState()

    def step(self, state: FakeState, token_id: int, position: int) -> torch.Tensor:
        assert position == len(state.tokens), "position must equal the consumed token count"
        if token_id == self.fail_on_token:
            raise RuntimeError("injected model failure")
        if self.step_delay:
            time.sleep(self.step_delay)
        state.tokens.append(token_id)
        with self._lock:
            self.steps.append((id(state), token_id))
        if ASSISTANT in state.tokens:
            produced = (
                len(state.tokens)
                - 1
                - max(index for index, token in enumerate(state.tokens) if token == ASSISTANT)
            )
            next_token = self.reply[produced] if produced < len(self.reply) else EOS
        else:
            next_token = EOS
        logits = torch.full((VOCAB,), -10.0)
        logits[next_token] = 10.0
        return logits


class FakeTokenizer:
    """Character-level tokenizer with a tiny chat template.

    The template renders earlier assistant turns as text but starts a new
    reply with the ASSISTANT marker, so a generated reply is never a prefix of
    the next prompt, which mirrors real chat templates that re-render history.
    """

    stop_token_ids = (EOS,)

    def encode(self, text: str) -> list[int]:
        return [ord(character) for character in text]

    def decode(self, token_ids: Sequence[int]) -> str:
        return "".join(chr(token) for token in token_ids if token not in (EOS, ASSISTANT))

    def encode_chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        add_generation_prompt: bool,
        template_kwargs: Mapping[str, Any],
    ) -> list[int]:
        self.last_template_kwargs = dict(template_kwargs)
        tokens = self.encode("".join(f"<{m['role']}>{m['content']}\n" for m in messages))
        return tokens + [ASSISTANT] if add_generation_prompt else tokens
