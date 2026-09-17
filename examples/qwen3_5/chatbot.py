from __future__ import annotations

import argparse
import importlib
import sys
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Any, NamedTuple, TextIO


DEFAULT_MODEL_ID = "Qwen/Qwen3.5-0.8B"
MIN_TRANSFORMERS_VERSION = "5.2.0"
DEFAULT_STOP_TOKEN_IDS = (248044, 248046)


class DependencyError(RuntimeError):
    """Raised when the installed runtime cannot load Qwen3.5."""


class TransformersRuntime(NamedTuple):
    auto_tokenizer: Any
    model_class: Any
    streamer_class: Any


class _CancellationCriteria:
    def __init__(self, cancelled: threading.Event) -> None:
        self._cancelled = cancelled

    def __call__(self, *args: Any, **kwargs: Any) -> bool:
        del args, kwargs
        return self._cancelled.is_set()


@dataclass(frozen=True)
class GenerationOptions:
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 20

    def as_kwargs(self) -> dict[str, Any]:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than zero")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in the interval (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")

        options: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            options.update(temperature=self.temperature, top_p=self.top_p, top_k=self.top_k)
        return options


def resolve_stop_token_ids(tokenizer: Any, model: Any) -> tuple[int, ...]:
    """Combine Qwen3.5's model-EOS and turn-terminator IDs."""

    token_ids = set(DEFAULT_STOP_TOKEN_IDS)

    def add(value: Any) -> None:
        if isinstance(value, int):
            token_ids.add(value)
        elif isinstance(value, (list, tuple, set)):
            token_ids.update(int(item) for item in value)

    add(getattr(tokenizer, "eos_token_id", None))
    add(getattr(getattr(model, "config", None), "eos_token_id", None))
    add(getattr(getattr(model, "generation_config", None), "eos_token_id", None))
    return tuple(sorted(token_ids))


def resolve_transformers(module: ModuleType | Any | None = None) -> TransformersRuntime:
    """Resolve the public Transformers classes while keeping module import testable."""

    if module is None:
        try:
            module = importlib.import_module("transformers")
        except ImportError as exc:
            raise DependencyError(
                "Transformers is required. Install it with "
                f"`python -m pip install 'transformers>={MIN_TRANSFORMERS_VERSION}'`."
            ) from exc

    missing = [
        name
        for name in ("AutoTokenizer", "Qwen3_5ForCausalLM", "TextIteratorStreamer")
        if not hasattr(module, name)
    ]
    if missing:
        version = getattr(module, "__version__", "unknown")
        raise DependencyError(
            f"Transformers {version} does not provide {', '.join(missing)}. "
            "Qwen3.5 support requires a newer release; upgrade with "
            f"`python -m pip install --upgrade 'transformers>={MIN_TRANSFORMERS_VERSION}'`."
        )

    return TransformersRuntime(
        module.AutoTokenizer,
        module.Qwen3_5ForCausalLM,
        module.TextIteratorStreamer,
    )


def select_device(requested: str, torch_module: ModuleType | Any) -> str:
    if requested == "auto":
        if torch_module.cuda.is_available():
            return "cuda"
        mps = getattr(torch_module.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    if requested == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch reports that CUDA is unavailable")
    if requested == "mps":
        mps = getattr(torch_module.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS was requested, but PyTorch reports that MPS is unavailable")
    return requested


def classify_input(text: str) -> str | None:
    command = text.strip().lower()
    if command == "/reset":
        return "reset"
    if command == "/exit":
        return "exit"
    return None


def _move_batch_to_device(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    if hasattr(batch, "to"):
        batch = batch.to(device)
    else:
        batch = {
            name: value.to(device) if hasattr(value, "to") else value
            for name, value in batch.items()
        }
    if not isinstance(batch, Mapping):
        raise TypeError("the tokenizer chat template must return a mapping of model inputs")
    return dict(batch)


def _validate_context_budget(inputs: Mapping[str, Any], model: Any, max_new_tokens: int) -> None:
    config = getattr(model, "config", None)
    max_position_embeddings = getattr(config, "max_position_embeddings", None)
    if max_position_embeddings is None:
        max_position_embeddings = getattr(
            getattr(config, "text_config", None), "max_position_embeddings", None
        )
    if not isinstance(max_position_embeddings, int) or max_position_embeddings <= 0:
        return

    input_ids = inputs.get("input_ids")
    shape = getattr(input_ids, "shape", None)
    if shape is None or len(shape) == 0:
        raise TypeError("the tokenizer chat template must return input_ids with a token dimension")
    prompt_tokens = int(shape[-1])
    if prompt_tokens + max_new_tokens > max_position_embeddings:
        raise ValueError(
            "conversation is too long: "
            f"{prompt_tokens} prompt tokens plus {max_new_tokens} requested tokens exceed "
            f"the model context window of {max_position_embeddings}"
        )


class ChatSession:
    """Owns chat history and performs one streamed, transactional model turn."""

    def __init__(
        self,
        tokenizer: Any,
        model: Any,
        streamer_factory: Callable[..., Iterable[str]],
        *,
        generation: GenerationOptions | None = None,
        system_prompt: str | None = None,
        thinking: bool = False,
    ) -> None:
        self._tokenizer = tokenizer
        self._model = model
        self._streamer_factory = streamer_factory
        self._generation = generation or GenerationOptions()
        self._system_prompt = system_prompt
        self._thinking = thinking
        self._stop_token_ids = resolve_stop_token_ids(tokenizer, model)
        self._messages: list[dict[str, str]] = []
        self.reset()

    @property
    def messages(self) -> tuple[dict[str, str], ...]:
        return tuple(message.copy() for message in self._messages)

    def reset(self) -> None:
        self._messages.clear()
        if self._system_prompt:
            self._messages.append({"role": "system", "content": self._system_prompt})

    def stream_reply(self, prompt: str, emit: Callable[[str], None]) -> str:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")

        self._messages.append({"role": "user", "content": prompt})
        try:
            generation_kwargs = self._generation.as_kwargs()
            encoded = self._tokenizer.apply_chat_template(
                self._messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=self._thinking,
            )
            inputs = _move_batch_to_device(encoded, self._model.device)
            _validate_context_budget(inputs, self._model, self._generation.max_new_tokens)
            streamer = self._streamer_factory(
                self._tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )
            generation_errors: list[BaseException] = []
            cancel_generation = threading.Event()

            def generate() -> None:
                try:
                    self._model.generate(
                        **inputs,
                        **generation_kwargs,
                        eos_token_id=list(self._stop_token_ids),
                        streamer=streamer,
                        stopping_criteria=[_CancellationCriteria(cancel_generation)],
                    )
                except BaseException as exc:  # Make worker failures visible to the caller.
                    generation_errors.append(exc)
                    try:
                        streamer.end()
                    except Exception:
                        pass

            worker = threading.Thread(target=generate, name="qwen3.5-generate", daemon=True)
            worker.start()

            chunks: list[str] = []
            try:
                for chunk in streamer:
                    chunks.append(chunk)
                    emit(chunk)
            finally:
                # Do not leave model.generate() running after an output consumer fails.
                cancel_generation.set()
                worker.join()

            if generation_errors:
                raise generation_errors[0]
            reply = "".join(chunks)
        except BaseException:
            self._messages.pop()
            raise

        self._messages.append({"role": "assistant", "content": reply})
        return reply


def load_session(
    model_id: str,
    *,
    device: str,
    local_files_only: bool,
    generation: GenerationOptions,
    system_prompt: str | None,
    thinking: bool,
) -> tuple[ChatSession, str]:
    runtime = resolve_transformers()
    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:
        raise DependencyError(
            "PyTorch is required; install it with `python -m pip install torch`."
        ) from exc

    selected_device = select_device(device, torch)
    tokenizer = runtime.auto_tokenizer.from_pretrained(
        model_id,
        local_files_only=local_files_only,
    )
    model = runtime.model_class.from_pretrained(
        model_id,
        dtype="auto",
        local_files_only=local_files_only,
    )
    model.to(selected_device)
    model.eval()

    return (
        ChatSession(
            tokenizer,
            model,
            runtime.streamer_class,
            generation=generation,
            system_prompt=system_prompt,
            thinking=thinking,
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
    parser = argparse.ArgumentParser(description="Text-only Qwen3.5 terminal chatbot")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
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
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="do not contact the Hugging Face Hub; use an already cached model",
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
        generation.as_kwargs()  # Validate before downloading a checkpoint.
        session, selected_device = load_session(
            args.model_id,
            device=args.device,
            local_files_only=args.local_files_only,
            generation=generation,
            system_prompt=args.system_prompt,
            thinking=args.thinking,
        )
        print(f"Loaded {args.model_id} on {selected_device}.", file=sys.stderr)
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
