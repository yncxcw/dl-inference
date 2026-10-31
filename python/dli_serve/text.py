"""Streaming text helpers: incremental detokenization and stop strings."""

from __future__ import annotations

from typing import Callable, Sequence


class IncrementalDecoder:
    """Stream text for a growing token sequence without re-decoding all of it.

    Each step decodes only the tokens since the last emitted boundary, so the
    cost per token stays constant. Text ending in U+FFFD (an incomplete
    multi-byte character) is held back until a later token completes it.
    """

    def __init__(self, decode: Callable[[Sequence[int]], str]) -> None:
        self._decode = decode
        self._token_ids: list[int] = []
        self._prefix_offset = 0
        self._read_offset = 0

    def _advance(self, *, final: bool) -> str:
        # Decoding from an earlier boundary gives the tokenizer the context it
        # needs for leading spaces and merged bytes at the window start.
        prefix = self._decode(self._token_ids[self._prefix_offset : self._read_offset])
        current = self._decode(self._token_ids[self._prefix_offset :])
        if not current.startswith(prefix):
            raise RuntimeError("tokenizer produced a non-prefix incremental decode")
        chunk = current[len(prefix) :]
        if not final and (not chunk or current.endswith("�")):
            return ""
        self._prefix_offset = self._read_offset
        self._read_offset = len(self._token_ids)
        return chunk

    def push(self, token_id: int) -> str:
        self._token_ids.append(token_id)
        return self._advance(final=False)

    def flush(self) -> str:
        return self._advance(final=True) if self._read_offset < len(self._token_ids) else ""


class StopStringFilter:
    """Cut streamed text at the first stop string, OpenAI style.

    Text that could still turn into a stop string is held back, so a stop
    string split across tokens is never partially emitted. The stop string
    itself is never emitted.
    """

    def __init__(self, stops: Sequence[str]) -> None:
        self._stops = tuple(stop for stop in stops if stop)
        self._buffer = ""

    def feed(self, text: str) -> tuple[str, bool]:
        """Return (text safe to emit, whether a stop string was reached)."""

        if not self._stops:
            return text, False
        self._buffer += text
        hits = [index for index in (self._buffer.find(stop) for stop in self._stops) if index >= 0]
        if hits:
            emitted, self._buffer = self._buffer[: min(hits)], ""
            return emitted, True
        hold = 0
        for length in range(min(len(self._buffer), max(map(len, self._stops)) - 1), 0, -1):
            if any(stop.startswith(self._buffer[-length:]) for stop in self._stops):
                hold = length
                break
        emitted = self._buffer[: len(self._buffer) - hold]
        self._buffer = self._buffer[len(emitted) :]
        return emitted, False

    def flush(self) -> str:
        emitted, self._buffer = self._buffer, ""
        return emitted
