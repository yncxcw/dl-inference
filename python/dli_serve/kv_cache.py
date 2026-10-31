"""Cross-request KV cache: snapshots of request state keyed by token prefix."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass
class _Entry:
    tokens: tuple[int, ...]
    state: Any
    logits: Any | None


@dataclass(frozen=True)
class CacheMatch:
    """A cache hit: `state` is a private copy the caller may advance."""

    tokens: int
    state: Any
    logits: Any | None


class PrefixCache:
    """LRU cache of `ExecutionState` snapshots keyed by the tokens they consumed.

    Request state cannot be rewound (Qwen3.5's linear-attention state is
    recurrent), so a prompt can only reuse an entry whose tokens are an exact
    prefix of it. Entries are snapshots that never advance: inserts and hits
    both copy, and a copy shares tensor storage until either side steps.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 0:
            raise ValueError("prefix cache capacity must be non-negative")
        self.capacity = capacity
        self._entries: OrderedDict[tuple[int, ...], _Entry] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.hit_tokens = 0

    def __len__(self) -> int:
        return len(self._entries)

    def lookup(self, prompt: Sequence[int]) -> CacheMatch | None:
        """Return the longest cached prefix of `prompt`, or None.

        An entry covering the whole prompt only counts when it stored the
        logits for its last token, since the caller needs them to sample.
        """

        prompt = tuple(prompt)
        best: _Entry | None = None
        for key, entry in self._entries.items():
            length = len(key)
            if length > len(prompt) or (length == len(prompt) and entry.logits is None):
                continue
            if best is not None and length <= len(best.tokens):
                continue
            if prompt[:length] == key:
                best = entry
        if best is None:
            self.misses += 1
            return None
        self._entries.move_to_end(best.tokens)
        self.hits += 1
        self.hit_tokens += len(best.tokens)
        return CacheMatch(len(best.tokens), best.state.copy(), best.logits)

    def insert(self, tokens: Sequence[int], state: Any, logits: Any | None = None) -> None:
        """Snapshot `state`, which has consumed exactly `tokens`."""

        if self.capacity == 0 or not tokens:
            return
        key = tuple(tokens)
        existing = self._entries.get(key)
        if existing is not None:
            if existing.logits is None:
                existing.logits = logits
            self._entries.move_to_end(key)
            return
        self._entries[key] = _Entry(key, state.copy(), logits)
        while len(self._entries) > self.capacity:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        self._entries.clear()

    def stats(self) -> dict[str, int]:
        return {
            "entries": len(self._entries),
            "capacity": self.capacity,
            "hits": self.hits,
            "misses": self.misses,
            "hit_tokens": self.hit_tokens,
        }
