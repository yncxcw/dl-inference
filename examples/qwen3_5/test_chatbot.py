from __future__ import annotations

import io
import json
import tempfile
import types
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, patch

from chatbot import (
    ARTIFACT_FORMAT,
    ChatSession,
    DependencyError,
    GenerationOptions,
    TransformersRuntime,
    classify_input,
    load_artifact_bundle,
    load_session,
    parse_args,
    resolve_dli,
    resolve_stop_token_ids,
    resolve_transformers,
    run_interactive,
    run_one_shot,
    select_device,
)


class FakeGenerationConfig:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class FakeBackend:
    def __init__(self, responses, *, max_context_tokens: int = 4096) -> None:
        self.responses = iter(responses)
        self.max_context_tokens = max_context_tokens
        self.calls = []
        self.reset_count = 0
        self.generate = Mock(side_effect=AssertionError("model.generate must never be called"))

    def reset(self) -> None:
        self.reset_count += 1


class FakeDliGeneration:
    GenerationConfig = FakeGenerationConfig

    def __init__(self) -> None:
        self.calls = []

    def generate_tokens(self, backend, prompt_ids, config):
        self.calls.append((backend, tuple(prompt_ids), config))
        response = next(backend.responses)
        if isinstance(response, BaseException):
            raise response
        for item in response:
            if isinstance(item, BaseException):
                raise item
            yield item


class FakeTokenizer:
    eos_token_id = 99

    def __init__(self) -> None:
        self.rendered_messages = []
        self.input_ids = [1, 2]
        self.decode_calls = []
        self.token_text = {
            10: "Hello",
            11: " there",
            12: "I remember.",
            99: "",
        }

    def apply_chat_template(self, messages, **kwargs):
        self.rendered_messages.append(deepcopy(messages))
        self.last_template_kwargs = kwargs
        return {"input_ids": [self.input_ids.copy()]}

    def decode(self, token_ids, **kwargs):
        self.decode_calls.append((tuple(token_ids), kwargs))
        return "".join(self.token_text[token_id] for token_id in token_ids)


class ChatSessionTest(unittest.TestCase):
    def make_session(self, responses, **kwargs):
        tokenizer = FakeTokenizer()
        backend = FakeBackend(
            responses,
            max_context_tokens=kwargs.pop("max_context_tokens", 4096),
        )
        dli_runtime = FakeDliGeneration()
        session = ChatSession(
            tokenizer,
            backend,
            dli_module=dli_runtime,
            **kwargs,
        )
        return session, tokenizer, backend, dli_runtime

    def test_retains_history_and_streams_incrementally_through_backend(self) -> None:
        session, tokenizer, backend, dli_runtime = self.make_session(
            [[10, 11, 99], [12]],
            system_prompt="Be concise.",
        )

        chunks = []
        self.assertEqual(session.stream_reply("Hi", chunks.append), "Hello there")
        self.assertEqual(chunks, ["Hello", " there"])
        self.assertEqual(session.stream_reply("What did I say?", lambda chunk: None), "I remember.")

        self.assertEqual(
            tokenizer.rendered_messages[1],
            [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello there"},
                {"role": "user", "content": "What did I say?"},
            ],
        )
        self.assertEqual(
            tokenizer.last_template_kwargs,
            {
                "tokenize": True,
                "add_generation_prompt": True,
                "return_dict": True,
                "enable_thinking": False,
            },
        )
        self.assertEqual(len(dli_runtime.calls), 2)
        self.assertIs(dli_runtime.calls[0][0], backend)
        self.assertEqual(dli_runtime.calls[0][1], (1, 2))
        config = dli_runtime.calls[0][2]
        self.assertEqual(config.kwargs["max_context_tokens"], 4096)
        self.assertEqual(config.kwargs["stop_token_ids"], (99, 248044, 248046))
        self.assertEqual(
            tokenizer.decode_calls[0][1],
            {"skip_special_tokens": True, "clean_up_tokenization_spaces": False},
        )
        backend.generate.assert_not_called()

    def test_failed_generation_rolls_back_turn_and_resets_backend(self) -> None:
        session, _, backend, _ = self.make_session(
            [[10, RuntimeError("generation failed")]],
            system_prompt="Stay helpful.",
        )
        chunks = []

        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            session.stream_reply("This fails", chunks.append)

        self.assertEqual(chunks, ["Hello"])
        self.assertEqual(
            session.messages,
            ({"role": "system", "content": "Stay helpful."},),
        )
        self.assertEqual(backend.reset_count, 1)
        backend.generate.assert_not_called()

    def test_output_failure_rolls_back_turn_without_a_worker_thread(self) -> None:
        session, _, backend, _ = self.make_session([[10, 11]])

        def fail_output(chunk: str) -> None:
            self.assertEqual(chunk, "Hello")
            raise RuntimeError("output failed")

        with self.assertRaisesRegex(RuntimeError, "output failed"):
            session.stream_reply("This turn fails", fail_output)

        self.assertEqual(session.messages, ())
        self.assertEqual(backend.reset_count, 1)
        backend.generate.assert_not_called()

    def test_multibyte_replacement_suffix_is_held_until_it_resolves(self) -> None:
        class MultibyteTokenizer(FakeTokenizer):
            def decode(self, token_ids, **kwargs):
                self.decode_calls.append((tuple(token_ids), kwargs))
                return {(1,): "\ufffd", (1, 2): "你"}[tuple(token_ids)]

        tokenizer = MultibyteTokenizer()
        backend = FakeBackend([[1, 2]])
        session = ChatSession(
            tokenizer,
            backend,
            dli_module=FakeDliGeneration(),
        )
        chunks = []

        self.assertEqual(session.stream_reply("Translate", chunks.append), "你")
        self.assertEqual(chunks, ["你"])
        backend.generate.assert_not_called()

    def test_reset_preserves_system_prompt_and_clears_backend(self) -> None:
        session, _, backend, _ = self.make_session([[10]], system_prompt="Stay helpful.")
        session.stream_reply("question", lambda chunk: None)

        session.reset()

        self.assertEqual(
            session.messages,
            ({"role": "system", "content": "Stay helpful."},),
        )
        self.assertEqual(backend.reset_count, 1)

    def test_thinking_mode_is_forwarded_to_chat_template(self) -> None:
        session, tokenizer, _, _ = self.make_session([[10]], thinking=True)
        session.stream_reply("Think about this", lambda chunk: None)
        self.assertTrue(tokenizer.last_template_kwargs["enable_thinking"])

    def test_rejects_turn_that_cannot_reserve_requested_output_tokens(self) -> None:
        session, tokenizer, backend, dli_runtime = self.make_session(
            [[10]],
            generation=GenerationOptions(max_new_tokens=3),
            max_context_tokens=8,
        )
        tokenizer.input_ids = list(range(6))

        with self.assertRaisesRegex(ValueError, "6 prompt tokens plus 3 requested tokens"):
            session.stream_reply("Too long", lambda chunk: None)

        self.assertEqual(dli_runtime.calls, [])
        self.assertEqual(session.messages, ())
        self.assertEqual(backend.reset_count, 1)


class FakeGraph:
    calls = []

    @classmethod
    def from_json_file(cls, path):
        cls.calls.append(path)
        return f"graph:{Path(path).name}"


class FakeEngine:
    pass


class FakeExecutionState:
    pass


class FakeEngineCausalLM:
    instances = []

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        self.max_context_tokens = kwargs["max_context_tokens"]
        self.reset_count = 0
        self.__class__.instances.append(self)

    def reset(self) -> None:
        self.reset_count += 1


class FakeAutoTokenizer:
    calls = []

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        cls.calls.append((model_id, kwargs))
        return FakeTokenizer()


class ArtifactRuntimeTest(unittest.TestCase):
    def setUp(self) -> None:
        FakeGraph.calls.clear()
        FakeEngineCausalLM.instances.clear()
        FakeAutoTokenizer.calls.clear()

    @staticmethod
    def write_bundle(directory: str, **overrides) -> Path:
        data = {
            "format": ARTIFACT_FORMAT,
            "prefill_graph": "graphs/prefill.json",
            "decode_graph": "graphs/decode.json",
            "weights": "weights/model.json",
            "max_context_tokens": 64,
            "model_id": "bundle/model",
        }
        data.update(overrides)
        path = Path(directory) / "bundle.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_bundle_resolves_relative_paths_and_defaults_input_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_bundle(directory)
            bundle = load_artifact_bundle(path)

        base = Path(directory).resolve()
        self.assertEqual(bundle.prefill_graph, base / "graphs/prefill.json")
        self.assertEqual(bundle.decode_graph, base / "graphs/decode.json")
        self.assertEqual(bundle.weights, base / "weights/model.json")
        self.assertEqual(bundle.max_context_tokens, 64)
        self.assertEqual(bundle.input_shape, (1,))
        self.assertEqual(bundle.model_id, "bundle/model")

    def test_bundle_rejects_wrong_format_absolute_paths_and_multitoken_shape(self) -> None:
        invalid_cases = (
            ({"format": "wrong"}, "format"),
            ({"weights": "/absolute/weights.json"}, "relative path"),
            ({"weights": "../outside/weights.json"}, "escapes the bundle directory"),
            ({"max_context_tokens": 0}, "positive integer"),
            ({"input_shape": [1, 2]}, "exactly one token"),
        )
        for overrides, message in invalid_cases:
            with self.subTest(overrides=overrides), tempfile.TemporaryDirectory() as directory:
                path = self.write_bundle(directory, **overrides)
                with self.assertRaisesRegex(ValueError, message):
                    load_artifact_bundle(path)

    def test_loader_builds_dli_backend_and_never_resolves_a_transformers_model(self) -> None:
        weight_calls = []
        weights = {"weight": object()}

        def load_weights(path, device):
            weight_calls.append((path, device))
            return weights

        fake_dli = types.SimpleNamespace(
            Engine=FakeEngine,
            EngineCausalLM=FakeEngineCausalLM,
            ExecutionState=FakeExecutionState,
            GenerationConfig=FakeGenerationConfig,
            Graph=FakeGraph,
            generate_tokens=Mock(),
            load_weights=load_weights,
        )
        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: True),
        )

        with tempfile.TemporaryDirectory() as directory:
            bundle_path = self.write_bundle(directory, input_shape=[1, 1])
            with (
                patch(
                    "chatbot.resolve_transformers",
                    return_value=TransformersRuntime(FakeAutoTokenizer),
                ),
                patch("chatbot.resolve_dli", return_value=fake_dli),
                patch("chatbot.importlib.import_module", return_value=fake_torch) as importer,
            ):
                session, device = load_session(
                    bundle_path,
                    "override/model",
                    device="auto",
                    local_files_only=True,
                    generation=GenerationOptions(max_new_tokens=4),
                    system_prompt="Be concise.",
                    thinking=False,
                )

            base = Path(directory).resolve()

        self.assertEqual(device, "cuda")
        self.assertEqual(session.model_id, "override/model")
        self.assertEqual(
            FakeAutoTokenizer.calls,
            [("override/model", {"local_files_only": True})],
        )
        importer.assert_called_once_with("torch")
        self.assertEqual(
            FakeGraph.calls,
            [str(base / "graphs/prefill.json"), str(base / "graphs/decode.json")],
        )
        self.assertEqual(weight_calls, [(str(base / "weights/model.json"), "cuda")])

        backend = FakeEngineCausalLM.instances[0]
        self.assertIsInstance(backend.args[0], FakeEngine)
        self.assertEqual(backend.args[1], "graph:decode.json")
        self.assertIs(backend.args[2], weights)
        self.assertEqual(backend.kwargs["prefill_graph"], "graph:prefill.json")
        self.assertIs(backend.kwargs["prefill_static_inputs"], weights)
        self.assertIsInstance(backend.kwargs["state"], FakeExecutionState)
        self.assertEqual(backend.kwargs["input_shape"], (1, 1))
        self.assertEqual(backend.kwargs["max_context_tokens"], 64)
        self.assertEqual(backend.kwargs["device"], "cuda")


class HelpersTest(unittest.TestCase):
    def test_commands_are_exact_and_case_insensitive(self) -> None:
        self.assertEqual(classify_input(" /RESET "), "reset")
        self.assertEqual(classify_input("/exit"), "exit")
        self.assertIsNone(classify_input("please /reset this"))

    def test_generation_options_create_dli_config(self) -> None:
        runtime = types.SimpleNamespace(GenerationConfig=FakeGenerationConfig)
        config = GenerationOptions(max_new_tokens=12, temperature=0).to_dli_config(
            runtime,
            stop_token_ids=(7,),
            max_context_tokens=32,
        )
        self.assertEqual(
            config.kwargs,
            {
                "max_new_tokens": 12,
                "temperature": 0,
                "top_p": 1.0,
                "top_k": 20,
                "stop_token_ids": (7,),
                "max_context_tokens": 32,
            },
        )

    def test_invalid_generation_options_fail_early(self) -> None:
        invalid = (
            (GenerationOptions(max_new_tokens=0), "greater than zero"),
            (GenerationOptions(temperature=-1), "non-negative"),
            (GenerationOptions(top_p=0), "interval"),
            (GenerationOptions(top_k=-1), "top_k"),
        )
        for options, message in invalid:
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, message):
                options.validate()

    def test_stop_ids_use_tokenizer_without_a_model(self) -> None:
        tokenizer = types.SimpleNamespace(eos_token_id=[7, 8])
        self.assertEqual(resolve_stop_token_ids(tokenizer), (7, 8, 248044, 248046))

    def test_transformers_runtime_requires_only_auto_tokenizer(self) -> None:
        module = types.SimpleNamespace(AutoTokenizer="tokenizer")
        self.assertEqual(resolve_transformers(module), ("tokenizer",))

        old_transformers = types.SimpleNamespace(__version__="5.1.0")
        with self.assertRaises(DependencyError) as raised:
            resolve_transformers(old_transformers)
        self.assertIn("AutoTokenizer", str(raised.exception))
        self.assertIn("5.1.0", str(raised.exception))

    def test_dli_runtime_is_imported_lazily_and_validated(self) -> None:
        complete = types.SimpleNamespace(
            Engine=object,
            EngineCausalLM=object,
            ExecutionState=object,
            GenerationConfig=object,
            Graph=object,
            generate_tokens=object,
            load_weights=object,
        )
        with patch("chatbot.importlib.import_module", return_value=complete) as importer:
            self.assertIs(resolve_dli(), complete)
        importer.assert_called_once_with("dli")

        with self.assertRaisesRegex(DependencyError, "EngineCausalLM"):
            resolve_dli(types.SimpleNamespace(Engine=object))

    def test_device_selection_supports_only_cpu_and_cuda(self) -> None:
        def fake_torch(cuda: bool):
            return types.SimpleNamespace(
                cuda=types.SimpleNamespace(is_available=lambda: cuda),
                backends=types.SimpleNamespace(
                    mps=types.SimpleNamespace(is_available=lambda: True)
                ),
            )

        self.assertEqual(select_device("auto", fake_torch(True)), "cuda")
        self.assertEqual(select_device("auto", fake_torch(False)), "cpu")
        self.assertEqual(select_device("cpu", fake_torch(False)), "cpu")
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            select_device("cuda", fake_torch(False))
        with self.assertRaisesRegex(ValueError, "only CPU and CUDA"):
            select_device("mps", fake_torch(False))

    def test_cli_requires_artifacts_and_accepts_generation_chat_options(self) -> None:
        with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            parse_args([])
        args = parse_args(
            [
                "--artifacts",
                "bundle.json",
                "--model-id",
                "override/model",
                "--device",
                "cpu",
                "--thinking",
                "--local-files-only",
            ]
        )
        self.assertEqual(args.artifacts, Path("bundle.json"))
        self.assertEqual(args.model_id, "override/model")
        self.assertEqual(args.device, "cpu")
        self.assertTrue(args.thinking)
        self.assertTrue(args.local_files_only)

    def test_one_shot_streams_answer_and_newline(self) -> None:
        session = types.SimpleNamespace(
            stream_reply=lambda prompt, emit: [emit("one"), emit(" shot")]
        )
        output = io.StringIO()

        run_one_shot(session, "hello", output=output)

        self.assertEqual(output.getvalue(), "one shot\n")

    def test_interactive_reset_and_exit(self) -> None:
        class Session:
            def __init__(self) -> None:
                self.resets = 0
                self.prompts = []

            def reset(self) -> None:
                self.resets += 1

            def stream_reply(self, prompt, emit) -> None:
                self.prompts.append(prompt)
                emit("answer")

        session = Session()
        inputs = iter(["/reset", "hello", "/exit"])
        output = io.StringIO()

        run_interactive(session, input_fn=lambda _: next(inputs), output=output)

        self.assertEqual(session.resets, 1)
        self.assertEqual(session.prompts, ["hello"])
        self.assertIn("Conversation reset.", output.getvalue())
        self.assertIn("Assistant: answer", output.getvalue())


if __name__ == "__main__":
    unittest.main()
