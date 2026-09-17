from __future__ import annotations

import io
import queue
import types
import unittest
from copy import deepcopy
from unittest.mock import Mock, patch

from chatbot import (
    ChatSession,
    DependencyError,
    GenerationOptions,
    classify_input,
    resolve_transformers,
    resolve_stop_token_ids,
    run_interactive,
    run_one_shot,
    select_device,
)


class FakeBatch(dict):
    def __init__(self, input_tokens: int) -> None:
        super().__init__(input_ids=FakeTensor(input_tokens))
        self.moved_to = None

    def to(self, device):
        self.moved_to = device
        return self


class FakeTensor:
    def __init__(self, input_tokens: int) -> None:
        self.shape = (1, input_tokens)


class FakeTokenizer:
    def __init__(self) -> None:
        self.rendered_messages = []
        self.batches = []
        self.input_tokens = 1

    def apply_chat_template(self, messages, **kwargs):
        self.rendered_messages.append(deepcopy(messages))
        self.last_template_kwargs = kwargs
        batch = FakeBatch(self.input_tokens)
        self.batches.append(batch)
        return batch


class FakeStreamer:
    _DONE = object()

    def __init__(self, tokenizer, **kwargs) -> None:
        self.tokenizer = tokenizer
        self.kwargs = kwargs
        self.items = queue.Queue()

    def put_text(self, text: str) -> None:
        self.items.put(text)

    def end(self) -> None:
        self.items.put(self._DONE)

    def __iter__(self):
        return self

    def __next__(self):
        item = self.items.get(timeout=2)
        if item is self._DONE:
            raise StopIteration
        return item


class ImmediateStreamer:
    def __init__(self, tokenizer, **kwargs) -> None:
        del tokenizer, kwargs

    def __iter__(self):
        return iter(("partial",))

    def end(self) -> None:
        pass


class FakeModel:
    device = "cpu"

    def __init__(self, responses) -> None:
        self.responses = iter(responses)
        self.generate_calls = []
        self.config = types.SimpleNamespace(max_position_embeddings=4096)

    def generate(self, **kwargs) -> None:
        self.generate_calls.append(kwargs.copy())
        streamer = kwargs["streamer"]
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        for chunk in response:
            streamer.put_text(chunk)
        streamer.end()


class ChatSessionTest(unittest.TestCase):
    def make_session(self, responses, **kwargs):
        tokenizer = FakeTokenizer()
        model = FakeModel(responses)
        session = ChatSession(tokenizer, model, FakeStreamer, **kwargs)
        return session, tokenizer, model

    def test_retains_history_and_streams_each_turn(self) -> None:
        session, tokenizer, model = self.make_session(
            [["Hello", " there"], ["I remember."]],
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
        self.assertEqual(tokenizer.batches[0].moved_to, "cpu")
        self.assertEqual(
            tokenizer.last_template_kwargs,
            {
                "tokenize": True,
                "add_generation_prompt": True,
                "return_dict": True,
                "return_tensors": "pt",
                "enable_thinking": False,
            },
        )
        self.assertTrue(model.generate_calls[0]["do_sample"])
        self.assertEqual(model.generate_calls[0]["eos_token_id"], [248044, 248046])
        self.assertEqual(len(model.generate_calls[0]["stopping_criteria"]), 1)
        self.assertTrue(model.generate_calls[0]["stopping_criteria"][0]())

    def test_failed_generation_does_not_commit_partial_turn(self) -> None:
        session, _, _ = self.make_session(
            [RuntimeError("generation failed")], system_prompt="Stay helpful."
        )

        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            session.stream_reply("This fails", lambda chunk: None)

        self.assertEqual(
            session.messages,
            ({"role": "system", "content": "Stay helpful."},),
        )

    def test_reset_preserves_system_prompt(self) -> None:
        session, _, _ = self.make_session([["answer"]], system_prompt="Stay helpful.")
        session.stream_reply("question", lambda chunk: None)

        session.reset()

        self.assertEqual(
            session.messages,
            ({"role": "system", "content": "Stay helpful."},),
        )

    def test_thinking_mode_is_forwarded_to_the_chat_template(self) -> None:
        session, tokenizer, _ = self.make_session([["reasoned answer"]], thinking=True)
        session.stream_reply("Think about this", lambda chunk: None)
        self.assertTrue(tokenizer.last_template_kwargs["enable_thinking"])

    def test_rejects_turn_that_cannot_reserve_requested_output_tokens(self) -> None:
        session, tokenizer, model = self.make_session(
            [["unused"]], generation=GenerationOptions(max_new_tokens=3)
        )
        tokenizer.input_tokens = 6
        model.config.max_position_embeddings = 8

        with self.assertRaisesRegex(ValueError, "6 prompt tokens plus 3 requested tokens"):
            session.stream_reply("Too long", lambda chunk: None)

        self.assertEqual(model.generate_calls, [])
        self.assertEqual(session.messages, ())

    def test_consumer_failure_cancels_joins_and_rolls_back_turn(self) -> None:
        tokenizer = FakeTokenizer()
        model = FakeModel([[]])
        session = ChatSession(tokenizer, model, ImmediateStreamer)
        worker = Mock()

        def fail_output(chunk: str) -> None:
            self.assertEqual(chunk, "partial")
            raise RuntimeError("output failed")

        with patch("chatbot.threading.Thread", return_value=worker) as thread_constructor:
            with self.assertRaisesRegex(RuntimeError, "output failed"):
                session.stream_reply("This turn fails", fail_output)

        worker.start.assert_called_once_with()
        worker.join.assert_called_once_with()
        thread_constructor.call_args.kwargs["target"]()
        self.assertTrue(model.generate_calls[0]["stopping_criteria"][0]())
        self.assertEqual(session.messages, ())


class HelpersTest(unittest.TestCase):
    def test_commands_are_exact_and_case_insensitive(self) -> None:
        self.assertEqual(classify_input(" /RESET "), "reset")
        self.assertEqual(classify_input("/exit"), "exit")
        self.assertIsNone(classify_input("please /reset this"))

    def test_greedy_generation_omits_sampling_only_options(self) -> None:
        self.assertEqual(
            GenerationOptions(max_new_tokens=12, temperature=0).as_kwargs(),
            {"max_new_tokens": 12, "do_sample": False},
        )

    def test_invalid_generation_options_fail_early(self) -> None:
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            GenerationOptions(max_new_tokens=0).as_kwargs()
        with self.assertRaisesRegex(ValueError, "non-negative"):
            GenerationOptions(temperature=-1).as_kwargs()
        with self.assertRaisesRegex(ValueError, "interval"):
            GenerationOptions(top_p=0).as_kwargs()
        with self.assertRaisesRegex(ValueError, "top_k"):
            GenerationOptions(top_k=-1).as_kwargs()

    def test_stop_ids_include_model_and_tokenizer_values(self) -> None:
        tokenizer = types.SimpleNamespace(eos_token_id=7)
        model = types.SimpleNamespace(
            config=types.SimpleNamespace(eos_token_id=8),
            generation_config=types.SimpleNamespace(eos_token_id=[9, 10]),
        )
        self.assertEqual(
            resolve_stop_token_ids(tokenizer, model),
            (7, 8, 9, 10, 248044, 248046),
        )

    def test_old_transformers_error_is_actionable(self) -> None:
        old_transformers = types.SimpleNamespace(
            __version__="5.1.0",
            AutoTokenizer=object(),
            TextIteratorStreamer=object(),
        )

        with self.assertRaises(DependencyError) as raised:
            resolve_transformers(old_transformers)

        message = str(raised.exception)
        self.assertIn("Qwen3_5ForCausalLM", message)
        self.assertIn("transformers>=5.2.0", message)
        self.assertIn("5.1.0", message)

    def test_resolves_public_transformers_apis(self) -> None:
        module = types.SimpleNamespace(
            AutoTokenizer="tokenizer",
            Qwen3_5ForCausalLM="model",
            TextIteratorStreamer="streamer",
        )

        self.assertEqual(
            resolve_transformers(module),
            ("tokenizer", "model", "streamer"),
        )

    def test_auto_device_prefers_cuda_then_mps_then_cpu(self) -> None:
        def fake_torch(cuda: bool, mps: bool):
            return types.SimpleNamespace(
                cuda=types.SimpleNamespace(is_available=lambda: cuda),
                backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps)),
            )

        self.assertEqual(select_device("auto", fake_torch(True, True)), "cuda")
        self.assertEqual(select_device("auto", fake_torch(False, True)), "mps")
        self.assertEqual(select_device("auto", fake_torch(False, False)), "cpu")

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
