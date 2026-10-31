"""DLI bundle runner and Hugging Face tokenizer adapter used by the server."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

ARTIFACT_FORMAT = "dli.autoregressive.v1"
# Qwen chat templates end assistant turns with <|im_end|>; <|endoftext|> ends
# documents. Both stop generation when the tokenizer defines them.
_CHAT_STOP_TOKENS = ("<|im_end|>", "<|endoftext|>")


@dataclass(frozen=True)
class ArtifactBundle:
    prefill_graph: Path
    decode_graph: Path
    weights: Path
    max_context_tokens: int
    input_shape: tuple[int, ...]
    model_id: str | None = None


def load_bundle(path: str | Path) -> ArtifactBundle:
    """Read and validate a `dli.autoregressive.v1` bundle description."""

    bundle_path = Path(path).expanduser().resolve()
    data = json.loads(bundle_path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping) or data.get("format") != ARTIFACT_FORMAT:
        raise ValueError(f"{bundle_path} is not a {ARTIFACT_FORMAT} bundle")
    base = bundle_path.parent

    def member(name: str) -> Path:
        value = data.get(name)
        if not isinstance(value, str) or not value.strip() or Path(value).is_absolute():
            raise ValueError(f"bundle field {name!r} must be a relative path")
        resolved = (base / value).resolve()
        if not resolved.is_relative_to(base):
            raise ValueError(f"bundle field {name!r} escapes the bundle directory")
        return resolved

    max_context_tokens = data.get("max_context_tokens")
    if (
        isinstance(max_context_tokens, bool)
        or not isinstance(max_context_tokens, int)
        or max_context_tokens <= 0
    ):
        raise ValueError("bundle max_context_tokens must be a positive integer")
    input_shape = data.get("input_shape", [1])
    if (
        not isinstance(input_shape, list)
        or not input_shape
        or any(isinstance(d, bool) or not isinstance(d, int) or d != 1 for d in input_shape)
    ):
        raise ValueError("bundle input_shape must describe exactly one token")
    model_id = data.get("model_id")
    return ArtifactBundle(
        prefill_graph=member("prefill_graph"),
        decode_graph=member("decode_graph"),
        weights=member("weights"),
        max_context_tokens=max_context_tokens,
        input_shape=tuple(input_shape),
        model_id=model_id if isinstance(model_id, str) and model_id.strip() else None,
    )


class DLIModelRunner:
    """Runs a bundle's one-token graphs over per-sequence `dli.ExecutionState`s.

    The prefill graph consumes position 0 and the decode graph every later
    position, matching `dli.EngineCausalLM`. Each step passes a fresh input
    tensor so no state can alias a buffer that a later step overwrites, which
    keeps prefix-cache snapshots isolated.
    """

    def __init__(
        self,
        bundle: ArtifactBundle,
        *,
        device: str | torch.device,
        input_name: str = "input_ids",
        output_name: str = "logits",
    ) -> None:
        self._dli = importlib.import_module("dli")
        self.device = torch.device(device)
        self.engine = self._dli.Engine()
        self.decode_graph = self._dli.Graph.from_json_file(str(bundle.decode_graph))
        self.prefill_graph = (
            self.decode_graph
            if bundle.prefill_graph == bundle.decode_graph
            else self._dli.Graph.from_json_file(str(bundle.prefill_graph))
        )
        self.weights = dict(self._dli.load_weights(str(bundle.weights), str(self.device)))
        self.max_context_tokens: int | None = bundle.max_context_tokens
        self._input_shape = bundle.input_shape
        self._input_name = input_name
        self._output_name = output_name

    def new_state(self) -> Any:
        return self._dli.ExecutionState()

    def step(self, state: Any, token_id: int, position: int) -> torch.Tensor:
        graph = self.prefill_graph if position == 0 else self.decode_graph
        inputs = dict(self.weights)
        inputs[self._input_name] = torch.full(
            self._input_shape, token_id, dtype=torch.int64, device=self.device
        )
        outputs = self.engine.run(graph, inputs, state=state, position_offset=position)
        if self._output_name not in outputs:
            raise KeyError(f"graph did not return the {self._output_name!r} output")
        return outputs[self._output_name].reshape(-1)


class HFTokenizer:
    """Adapts a Hugging Face tokenizer to the server's tokenizer interface."""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        stop_ids: set[int] = set()
        eos = getattr(tokenizer, "eos_token_id", None)
        if isinstance(eos, int):
            stop_ids.add(eos)
        elif isinstance(eos, (list, tuple, set)):
            stop_ids.update(int(token_id) for token_id in eos)
        unknown = getattr(tokenizer, "unk_token_id", None)
        for token in _CHAT_STOP_TOKENS:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if isinstance(token_id, int) and token_id >= 0 and token_id != unknown:
                stop_ids.add(token_id)
        self.stop_token_ids = tuple(sorted(stop_ids))

    @classmethod
    def from_pretrained(cls, model_id: str, *, local_files_only: bool = False) -> HFTokenizer:
        transformers = importlib.import_module("transformers")
        return cls(
            transformers.AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
        )

    def encode(self, text: str) -> list[int]:
        return [
            int(token_id) for token_id in self._tokenizer.encode(text, add_special_tokens=False)
        ]

    def encode_chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        add_generation_prompt: bool,
        template_kwargs: Mapping[str, Any],
    ) -> list[int]:
        encoded = self._tokenizer.apply_chat_template(
            [dict(message) for message in messages],
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_dict=True,
            **template_kwargs,
        )
        token_ids = encoded["input_ids"]
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if token_ids and isinstance(token_ids[0], (list, tuple)):
            token_ids = token_ids[0]
        return [int(token_id) for token_id in token_ids]

    def decode(self, token_ids: Sequence[int]) -> str:
        return self._tokenizer.decode(
            list(token_ids), skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
