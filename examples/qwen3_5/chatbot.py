from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, NamedTuple, TextIO


DEFAULT_MODEL_ID = "Qwen/Qwen3.5-0.8B"
MIN_TRANSFORMERS_VERSION = "5.2.0"
DEFAULT_STOP_TOKEN_IDS = (248044, 248046)
ARTIFACT_FORMAT = "dli.autoregressive.v1"


class DependencyError(RuntimeError):
    """Raised when the installed runtime cannot load the chatbot."""


class TransformersRuntime(NamedTuple):
    auto_tokenizer: Any


@dataclass(frozen=True)
class ArtifactBundle:
    prefill_graph: Path
    decode_graph: Path
    weights: Path
    max_context_tokens: int
    input_shape: tuple[int, ...]
    model_id: str | None = None


@dataclass(frozen=True)
class GenerationOptions:
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 20

    def validate(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than zero")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in the interval (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")

    def to_dli_config(
        self,
        dli_module: Any,
        *,
        stop_token_ids: tuple[int, ...],
        max_context_tokens: int | None,
    ) -> Any:
        self.validate()
        return dli_module.GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            stop_token_ids=stop_token_ids,
            max_context_tokens=max_context_tokens,
        )


def resolve_transformers(module: ModuleType | Any | None = None) -> TransformersRuntime:
    """Resolve only the tokenizer API needed at chat runtime."""

    if module is None:
        try:
            module = importlib.import_module("transformers")
        except ImportError as exc:
            raise DependencyError(
                "Transformers is required for the tokenizer. Install it with "
                f"`python -m pip install 'transformers>={MIN_TRANSFORMERS_VERSION}'`."
            ) from exc

    if not hasattr(module, "AutoTokenizer"):
        version = getattr(module, "__version__", "unknown")
        raise DependencyError(
            f"Transformers {version} does not provide AutoTokenizer; upgrade with "
            f"`python -m pip install --upgrade 'transformers>={MIN_TRANSFORMERS_VERSION}'`."
        )
    return TransformersRuntime(module.AutoTokenizer)


def resolve_dli(module: ModuleType | Any | None = None) -> Any:
    """Import the DLI runtime lazily so helper tests remain fully offline."""

    if module is None:
        try:
            module = importlib.import_module("dli")
        except ImportError as exc:
            raise DependencyError(
                "The DLI Python runtime is required; build or install the dl-inference wheel."
            ) from exc

    required = (
        "Engine",
        "EngineCausalLM",
        "ExecutionState",
        "GenerationConfig",
        "Graph",
        "generate_tokens",
        "load_weights",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise DependencyError(f"the DLI Python runtime is missing: {', '.join(missing)}")
    return module


def _relative_artifact_path(data: Mapping[str, Any], name: str, base: Path) -> Path:
    value = data.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"artifact bundle field {name!r} must be a relative path")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"artifact bundle field {name!r} must be a relative path")
    resolved_base = base.resolve()
    resolved = (resolved_base / relative).resolve()
    if not resolved.is_relative_to(resolved_base):
        raise ValueError(f"artifact bundle field {name!r} escapes the bundle directory")
    return resolved


def load_artifact_bundle(path: str | Path) -> ArtifactBundle:
    bundle_path = Path(path).expanduser().resolve()
    data = json.loads(bundle_path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("artifact bundle must contain a JSON object")
    if data.get("format") != ARTIFACT_FORMAT:
        raise ValueError(f"artifact bundle format must be {ARTIFACT_FORMAT!r}")

    max_context_tokens = data.get("max_context_tokens")
    if (
        isinstance(max_context_tokens, bool)
        or not isinstance(max_context_tokens, int)
        or max_context_tokens <= 0
    ):
        raise ValueError("artifact bundle max_context_tokens must be a positive integer")

    raw_input_shape = data.get("input_shape", [1])
    if not isinstance(raw_input_shape, (list, tuple)) or not raw_input_shape:
        raise ValueError("artifact bundle input_shape must be a non-empty array")
    if any(
        isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0
        for dimension in raw_input_shape
    ):
        raise ValueError("artifact bundle input_shape must contain positive integers")
    input_elements = 1
    for dimension in raw_input_shape:
        input_elements *= dimension
    if input_elements != 1:
        raise ValueError("artifact bundle input_shape must describe exactly one token")

    model_id = data.get("model_id")
    if model_id is not None and (not isinstance(model_id, str) or not model_id.strip()):
        raise ValueError("artifact bundle model_id must be a non-empty string")

    base = bundle_path.parent
    return ArtifactBundle(
        prefill_graph=_relative_artifact_path(data, "prefill_graph", base),
        decode_graph=_relative_artifact_path(data, "decode_graph", base),
        weights=_relative_artifact_path(data, "weights", base),
        max_context_tokens=max_context_tokens,
        input_shape=tuple(raw_input_shape),
        model_id=model_id,
    )


def resolve_stop_token_ids(tokenizer: Any) -> tuple[int, ...]:
    """Combine Qwen3.5 turn terminators with tokenizer EOS IDs."""

    token_ids = set(DEFAULT_STOP_TOKEN_IDS)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_token_id, int):
        token_ids.add(eos_token_id)
    elif isinstance(eos_token_id, (list, tuple, set)):
        token_ids.update(int(item) for item in eos_token_id)
    return tuple(sorted(token_ids))


def select_device(requested: str, torch_module: ModuleType | Any) -> str:
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError("DLI supports only CPU and CUDA devices")
    if requested == "auto":
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch reports that CUDA is unavailable")
    return requested


def classify_input(text: str) -> str | None:
    command = text.strip().lower()
    if command == "/reset":
        return "reset"
    if command == "/exit":
        return "exit"
    return None


def _extract_input_ids(encoded: Any) -> tuple[int, ...]:
    if not isinstance(encoded, Mapping):
        raise TypeError("the tokenizer chat template must return a mapping of model inputs")
    if "input_ids" not in encoded:
        raise KeyError("the tokenizer chat template did not return input_ids")

    input_ids = encoded["input_ids"]
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if (
        isinstance(input_ids, (list, tuple))
        and input_ids
        and isinstance(input_ids[0], (list, tuple))
    ):
        if len(input_ids) != 1:
            raise ValueError("generation currently supports batch size one")
        input_ids = input_ids[0]
    if isinstance(input_ids, (str, bytes)) or not isinstance(input_ids, Sequence):
        raise TypeError("the tokenizer chat template must return a sequence of input_ids")

    prompt = tuple(int(token_id) for token_id in input_ids)
    if not prompt:
        raise ValueError("the tokenizer chat template returned an empty prompt")
    if any(token_id < 0 for token_id in prompt):
        raise ValueError("the tokenizer chat template returned a negative token ID")
    return prompt


def _validate_context_budget(
    input_ids: Sequence[int],
    max_context_tokens: int | None,
    max_new_tokens: int,
) -> None:
    if max_context_tokens is None:
        return
    prompt_tokens = len(input_ids)
    if prompt_tokens + max_new_tokens > max_context_tokens:
        raise ValueError(
            "conversation is too long: "
            f"{prompt_tokens} prompt tokens plus {max_new_tokens} requested tokens exceed "
            f"the model context window of {max_context_tokens}"
        )


class ChatSession:
    """Own chat history and perform transactional turns through an autoregressive backend."""

    def __init__(
        self,
        tokenizer: Any,
        backend: Any,
        *,
        generation: GenerationOptions | None = None,
        system_prompt: str | None = None,
        thinking: bool = False,
        dli_module: Any | None = None,
        model_id: str | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        self._backend = backend
        self._generation = generation or GenerationOptions()
        self._system_prompt = system_prompt
        self._thinking = thinking
        self._dli_module = dli_module
        self._stop_token_ids = resolve_stop_token_ids(tokenizer)
        self._messages: list[dict[str, str]] = []
        self.model_id = model_id
        self._reset_messages()

    @property
    def messages(self) -> tuple[dict[str, str], ...]:
        return tuple(message.copy() for message in self._messages)

    def _reset_messages(self) -> None:
        self._messages.clear()
        if self._system_prompt:
            self._messages.append({"role": "system", "content": self._system_prompt})

    def _reset_backend(self) -> None:
        reset = getattr(self._backend, "reset", None)
        if callable(reset):
            reset()

    def reset(self) -> None:
        self._reset_messages()
        self._reset_backend()

    def _runtime(self) -> Any:
        if self._dli_module is None:
            self._dli_module = resolve_dli()
        return self._dli_module

    def stream_reply(self, prompt: str, emit: Callable[[str], None]) -> str:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")

        self._messages.append({"role": "user", "content": prompt})
        try:
            encoded = self._tokenizer.apply_chat_template(
                self._messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                enable_thinking=self._thinking,
            )
            input_ids = _extract_input_ids(encoded)
            max_context_tokens = getattr(self._backend, "max_context_tokens", None)
            _validate_context_budget(
                input_ids,
                max_context_tokens,
                self._generation.max_new_tokens,
            )

            runtime = self._runtime()
            config = self._generation.to_dli_config(
                runtime,
                stop_token_ids=self._stop_token_ids,
                max_context_tokens=max_context_tokens,
            )
            generated_ids: list[int] = []
            reply = ""
            for token_id in runtime.generate_tokens(self._backend, input_ids, config):
                generated_ids.append(int(token_id))
                decoded = self._tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                if not isinstance(decoded, str):
                    raise TypeError("tokenizer.decode must return a string")
                stable = decoded.split("\ufffd", 1)[0]
                if not stable.startswith(reply):
                    raise RuntimeError("tokenizer produced a non-prefix incremental decode")
                chunk = stable[len(reply) :]
                if chunk:
                    emit(chunk)
                reply = stable

            decoded = self._tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if not isinstance(decoded, str):
                raise TypeError("tokenizer.decode must return a string")
            if not decoded.startswith(reply):
                raise RuntimeError("tokenizer produced a non-prefix final decode")
            chunk = decoded[len(reply) :]
            if chunk:
                emit(chunk)
            reply = decoded
        except BaseException:
            self._messages.pop()
            try:
                self._reset_backend()
            except Exception:
                pass
            raise

        self._messages.append({"role": "assistant", "content": reply})
        return reply


def load_session(
    artifacts: str | Path,
    model_id: str | None = None,
    *,
    device: str,
    local_files_only: bool,
    generation: GenerationOptions,
    system_prompt: str | None,
    thinking: bool,
) -> tuple[ChatSession, str]:
    bundle = load_artifact_bundle(artifacts)
    resolved_model_id = model_id or bundle.model_id or DEFAULT_MODEL_ID
    transformers_runtime = resolve_transformers()
    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:
        raise DependencyError(
            "PyTorch is required; install it with `python -m pip install torch`."
        ) from exc
    dli = resolve_dli()

    selected_device = select_device(device, torch)
    tokenizer = transformers_runtime.auto_tokenizer.from_pretrained(
        resolved_model_id,
        local_files_only=local_files_only,
    )
    prefill_graph = dli.Graph.from_json_file(str(bundle.prefill_graph))
    decode_graph = dli.Graph.from_json_file(str(bundle.decode_graph))
    weights = dli.load_weights(str(bundle.weights), selected_device)
    engine = dli.Engine()
    state = dli.ExecutionState()
    backend = dli.EngineCausalLM(
        engine,
        decode_graph,
        weights,
        device=selected_device,
        max_context_tokens=bundle.max_context_tokens,
        state=state,
        prefill_graph=prefill_graph,
        prefill_static_inputs=weights,
        input_shape=bundle.input_shape,
    )

    return (
        ChatSession(
            tokenizer,
            backend,
            generation=generation,
            system_prompt=system_prompt,
            thinking=thinking,
            dli_module=dli,
            model_id=resolved_model_id,
        ),
        selected_device,
    )


def _write_stream(chunk: str, output: TextIO) -> None:
    output.write(chunk)
    output.flush()


def run_one_shot(session: ChatSession, prompt: str, *, output: TextIO = sys.stdout) -> None:
    session.stream_reply(prompt, lambda chunk: _write_stream(chunk, output))
    output.write("\n")
    output.flush()


def run_interactive(
    session: ChatSession,
    *,
    input_fn: Callable[[str], str] = input,
    output: TextIO = sys.stdout,
) -> None:
    output.write("Commands: /reset clears the conversation; /exit quits.\n")
    output.flush()
    while True:
        try:
            prompt = input_fn("You: ")
        except EOFError:
            output.write("\n")
            return

        command = classify_input(prompt)
        if command == "exit":
            return
        if command == "reset":
            session.reset()
            output.write("Conversation reset.\n")
            output.flush()
            continue
        if not prompt.strip():
            continue

        output.write("Assistant: ")
        output.flush()
        session.stream_reply(prompt, lambda chunk: _write_stream(chunk, output))
        output.write("\n")
        output.flush()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Text-only Qwen3.5 DLI terminal chatbot")
    parser.add_argument(
        "--artifacts",
        type=Path,
        required=True,
        help="dli.autoregressive.v1 artifact bundle JSON",
    )
    parser.add_argument("--model-id", help="override the tokenizer model ID in the bundle")
    parser.add_argument("--prompt", help="run one prompt and exit instead of starting a chat")
    parser.add_argument("--system-prompt")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="sampling temperature; use 0 for greedy decoding",
    )
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="enable Qwen3.5 thinking mode (the 0.8B model may loop in this mode)",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="do not contact the Hugging Face Hub; use an already cached tokenizer",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        generation = GenerationOptions(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
        generation.validate()
        session, selected_device = load_session(
            args.artifacts,
            args.model_id,
            device=args.device,
            local_files_only=args.local_files_only,
            generation=generation,
            system_prompt=args.system_prompt,
            thinking=args.thinking,
        )
        print(f"Loaded {session.model_id} DLI artifacts on {selected_device}.", file=sys.stderr)
        if args.prompt is not None:
            run_one_shot(session, args.prompt)
        else:
            run_interactive(session)
        return 0
    except (DependencyError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
