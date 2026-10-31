from __future__ import annotations

import importlib.util
import json

from dli_serve import (
    Failed,
    Finished,
    GenerationRequest,
    PrefixCache,
    SamplingParams,
    Scheduler,
    TextDelta,
)
from dli_serve.text import IncrementalDecoder, StopStringFilter
from serve_fakes import ASSISTANT, EOS, FakeRunner, FakeState, FakeTokenizer

GREEDY = SamplingParams(max_tokens=64, temperature=0.0, stop_token_ids=(EOS,))


def make_scheduler(runner: FakeRunner | None = None, **kwargs) -> Scheduler:
    return Scheduler(runner or FakeRunner(), FakeTokenizer().decode, **kwargs)


def run_request(scheduler: Scheduler, request: GenerationRequest) -> tuple[str, Finished]:
    events: list = []
    scheduler.submit(request, events.append)
    while scheduler.has_work():
        scheduler.step_round()
    finished = events[-1]
    assert isinstance(finished, Finished), events
    return "".join(event.text for event in events if isinstance(event, TextDelta)), finished


def chat_prompt(*turns: str) -> tuple[int, ...]:
    messages = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": text}
        for index, text in enumerate(turns)
    ]
    return tuple(
        FakeTokenizer().encode_chat(messages, add_generation_prompt=True, template_kwargs={})
    )


def test_prefix_cache_keeps_longest_exact_prefix_with_lru_eviction() -> None:
    cache = PrefixCache(capacity=2)
    assert cache.lookup([1, 2, 3]) is None
    cache.insert([1], FakeState([1]))
    cache.insert([1, 2], FakeState([1, 2]))
    match = cache.lookup([1, 2, 3])
    assert match is not None and match.tokens == 2 and match.state.tokens == [1, 2]
    # Hits hand out copies, so advancing one never changes the snapshot.
    match.state.tokens.append(3)
    assert cache.lookup([1, 2, 3]).state.tokens == [1, 2]
    # A full-prompt entry only counts when it stored logits to sample from.
    assert cache.lookup([1, 2]).tokens == 1
    cache.insert([1, 2], FakeState([1, 2]), logits="logits")
    assert cache.lookup([1, 2]).tokens == 2
    cache.insert([7], FakeState([7]))  # evicts the least recently used entry [1]
    assert cache.lookup([1, 9]) is None
    assert cache.stats()["entries"] == 2
    assert PrefixCache(capacity=0).lookup([1]) is None


def test_text_helpers_hold_back_partial_characters_and_stop_strings() -> None:
    pieces = {(): "", (1,): "�", (1, 2): "你", (2,): "�", (2, 3): "�x"}
    decoder = IncrementalDecoder(lambda ids: pieces.get(tuple(ids), "?"))
    assert decoder.push(1) == ""
    assert decoder.push(2) == "你"

    stops = StopStringFilter(["END"])
    assert stops.feed("abc E") == ("abc ", False)
    assert stops.feed("N") == ("", False)
    assert stops.feed("Dx") == ("", True)
    assert StopStringFilter(["END"]).feed("no stop E") == ("no stop ", False)
    assert StopStringFilter([]).feed("text") == ("text", False)


def test_scheduler_streams_reply_and_reports_usage() -> None:
    scheduler = make_scheduler()
    prompt = chat_prompt("Hi")
    text, finished = run_request(scheduler, GenerationRequest(prompt, GREEDY))
    assert text == "Hello there!"
    assert finished == Finished("stop", len(prompt), len("Hello there!") + 1, 0)
    assert scheduler.stats()["finished"] == 1


def test_identical_prompt_reuses_the_whole_prompt_snapshot() -> None:
    runner = FakeRunner()
    scheduler = make_scheduler(runner)
    prompt = chat_prompt("Hi")
    run_request(scheduler, GenerationRequest(prompt, GREEDY))
    steps_before = len(runner.steps)
    text, finished = run_request(scheduler, GenerationRequest(prompt, GREEDY))
    assert text == "Hello there!"
    assert finished.cached_tokens == len(prompt)
    # No prefill at all: one step per generated token except the final EOS.
    assert len(runner.steps) - steps_before == finished.completion_tokens - 1


def test_next_chat_turn_reuses_the_history_snapshot() -> None:
    runner = FakeRunner()
    scheduler = make_scheduler(runner)
    tokenizer = FakeTokenizer()
    first = [{"role": "user", "content": "Hi"}]
    history = tokenizer.encode_chat(first, add_generation_prompt=False, template_kwargs={})
    prompt = chat_prompt("Hi")
    run_request(scheduler, GenerationRequest(prompt, GREEDY, snapshot_points=(len(history),)))

    second = chat_prompt("Hi", "Hello there!", "And now?")
    steps_before = len(runner.steps)
    _, finished = run_request(scheduler, GenerationRequest(second, GREEDY))
    assert finished.cached_tokens == len(history)
    prefill_steps = len(runner.steps) - steps_before - (finished.completion_tokens - 1)
    assert prefill_steps == len(second) - len(history)


def test_round_robin_interleaves_concurrent_requests() -> None:
    runner = FakeRunner()
    scheduler = make_scheduler(runner, max_num_seqs=2)
    events: dict[str, list] = {"a": [], "b": []}
    scheduler.submit(GenerationRequest(chat_prompt("first"), GREEDY), events["a"].append)
    scheduler.submit(GenerationRequest(chat_prompt("second"), GREEDY), events["b"].append)
    scheduler.step_round()
    scheduler.step_round()
    first_states = [state for state, _ in runner.steps[:4]]
    assert first_states[0] == first_states[2] != first_states[1] == first_states[3]
    while scheduler.has_work():
        scheduler.step_round()
    for name in events:
        assert "".join(e.text for e in events[name] if isinstance(e, TextDelta)) == "Hello there!"


def test_requests_beyond_max_num_seqs_wait_their_turn() -> None:
    scheduler = make_scheduler(max_num_seqs=1)
    results: list[list] = [[], []]
    for result in results:
        scheduler.submit(GenerationRequest(chat_prompt("hi"), GREEDY), result.append)
    scheduler.step_round()
    assert scheduler.stats()["active"] == 1 and scheduler.stats()["pending"] == 1
    while scheduler.has_work():
        scheduler.step_round()
    assert all(isinstance(result[-1], Finished) for result in results)


def test_stop_strings_and_length_limits() -> None:
    scheduler = make_scheduler()
    prompt = chat_prompt("Hi")
    stop = SamplingParams(max_tokens=64, temperature=0.0, stop=("there",), stop_token_ids=(EOS,))
    assert run_request(scheduler, GenerationRequest(prompt, stop)) == (
        "Hello ",
        Finished("stop", len(prompt), len("Hello there"), 0),
    )
    short = SamplingParams(max_tokens=5, temperature=0.0, stop_token_ids=(EOS,))
    text, finished = run_request(make_scheduler(), GenerationRequest(prompt, short))
    assert (text, finished.finish_reason) == ("Hello", "length")

    tight = make_scheduler(FakeRunner(max_context_tokens=len(prompt) + 3))
    text, finished = run_request(tight, GenerationRequest(prompt, GREEDY))
    assert (text, finished.finish_reason) == ("Hell", "length")


def test_cancel_and_failures_only_affect_their_request() -> None:
    runner = FakeRunner()
    scheduler = make_scheduler(runner, max_num_seqs=2)
    cancelled: list = []
    handle = scheduler.submit(GenerationRequest(chat_prompt("one"), GREEDY), cancelled.append)
    scheduler.step_round()
    handle.cancel()
    scheduler.step_round()
    assert cancelled[-1].finish_reason == "abort"

    runner.fail_on_token = ord("!")
    failing: list = []
    healthy: list = []
    scheduler.submit(GenerationRequest(chat_prompt("bad!"), GREEDY), failing.append)
    scheduler.submit(GenerationRequest(chat_prompt("good"), GREEDY), healthy.append)
    while scheduler.has_work():
        runner.fail_on_token = ord("!") if failing == [] else None
        scheduler.step_round()
    assert isinstance(failing[-1], Failed)
    assert isinstance(healthy[-1], Finished) and healthy[-1].finish_reason == "stop"
    assert scheduler.stats()["failed"] == 1


def test_submit_validates_prompt_and_sampling() -> None:
    scheduler = make_scheduler(FakeRunner(max_context_tokens=4))
    for request in (
        GenerationRequest((), GREEDY),
        GenerationRequest((1, 2, 3, 4, 5), GREEDY),
        GenerationRequest((1,), SamplingParams(max_tokens=0)),
        GenerationRequest((1,), SamplingParams(max_tokens=4, top_p=0.0)),
    ):
        try:
            scheduler.submit(request, lambda event: None)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted an invalid request: {request}")


def test_threaded_scheduler_serves_requests_and_shuts_down() -> None:
    scheduler = make_scheduler()
    scheduler.start()
    done: list = []
    import threading

    finished = threading.Event()

    def emit(event) -> None:
        done.append(event)
        if not isinstance(event, TextDelta):
            finished.set()

    scheduler.submit(GenerationRequest(chat_prompt("Hi"), GREEDY), emit)
    assert finished.wait(10)
    scheduler.shutdown()
    assert isinstance(done[-1], Finished)
    try:
        scheduler.submit(GenerationRequest(chat_prompt("late"), GREEDY), emit)
    except RuntimeError:
        pass
    else:
        raise AssertionError("a stopped scheduler accepted work")


# HTTP API tests need FastAPI and its test client (httpx).


def make_client(**kwargs):
    from fastapi.testclient import TestClient

    from dli_serve.api import create_app

    scheduler = Scheduler(
        FakeRunner(max_context_tokens=kwargs.pop("max_context_tokens", 512)),
        FakeTokenizer().decode,
    )
    app = create_app(scheduler, FakeTokenizer(), model_name="fake-model", **kwargs)
    return TestClient(app), scheduler


def sse_events(response) -> list:
    events = []
    for line in response.text.splitlines():
        if line.startswith("data: "):
            data = line[len("data: ") :]
            events.append(data if data == "[DONE]" else json.loads(data))
    return events


def test_api_models_health_stats_and_ui() -> None:
    client, _ = make_client()
    with client:
        models = client.get("/v1/models").json()
        assert models["object"] == "list" and models["data"][0]["id"] == "fake-model"
        assert client.get("/v1/models/fake-model").json()["max_model_len"] == 512
        assert client.get("/v1/models/other").status_code == 404
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/stats").json()["model"] == "fake-model"
        page = client.get("/")
        assert page.status_code == 200 and "<title>" in page.text
        assert client.get("/static/app.js").status_code == 200


def test_api_chat_completion_and_cached_usage_across_turns() -> None:
    client, _ = make_client()
    with client:
        body = {"model": "fake-model", "messages": [{"role": "user", "content": "Hi"}]}
        first = client.post("/v1/chat/completions", json={**body, "temperature": 0}).json()
        assert first["object"] == "chat.completion"
        assert first["choices"][0]["message"] == {"role": "assistant", "content": "Hello there!"}
        assert first["choices"][0]["finish_reason"] == "stop"
        usage = first["usage"]
        assert (
            usage["completion_tokens"] == 13
            and usage["prompt_tokens_details"]["cached_tokens"] == 0
        )

        follow_up = {
            "model": "fake-model",
            "temperature": 0,
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello there!"},
                {"role": "user", "content": [{"type": "text", "text": "Again"}]},
            ],
        }
        second = client.post("/v1/chat/completions", json=follow_up).json()
        cached = second["usage"]["prompt_tokens_details"]["cached_tokens"]
        assert cached == len("<user>Hi\n"), second["usage"]


def test_api_chat_streaming_follows_the_openai_chunk_format() -> None:
    client, _ = make_client()
    with client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "fake-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
                "temperature": 0,
                "stream_options": {"include_usage": True},
            },
        )
        assert response.headers["content-type"].startswith("text/event-stream")
        events = sse_events(response)
    assert events[-1] == "[DONE]"
    chunks = events[:-1]
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
    assert text == "Hello there!"
    assert chunks[-2]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["choices"] == [] and chunks[-1]["usage"]["completion_tokens"] == 13


def test_api_completions_plain_and_streaming() -> None:
    client, _ = make_client()
    prompt = "Say:" + chr(ASSISTANT)
    with client:
        plain = client.post(
            "/v1/completions", json={"model": "fake-model", "prompt": prompt, "temperature": 0}
        ).json()
        assert (
            plain["object"] == "text_completion" and plain["choices"][0]["text"] == "Hello there!"
        )
        streamed = sse_events(
            client.post(
                "/v1/completions",
                json={"model": "fake-model", "prompt": prompt, "stream": True, "max_tokens": 5},
            )
        )
    assert "".join(e["choices"][0]["text"] for e in streamed[:-1]) == "Hello"
    assert streamed[-2]["choices"][0]["finish_reason"] == "length"


def test_api_errors_use_the_openai_error_format() -> None:
    client, _ = make_client(max_context_tokens=16)
    base = {"model": "fake-model", "messages": [{"role": "user", "content": "Hi"}]}
    with client:
        cases = [
            ({**base, "n": 2}, 400),
            ({**base, "tools": [{"type": "function"}]}, 400),
            ({**base, "temperature": 5}, 400),
            ({**base, "model": "missing"}, 404),
            ({**base, "messages": [{"role": "user", "content": "x" * 40}]}, 400),
            ({**base, "messages": []}, 400),
        ]
        for body, status in cases:
            response = client.post("/v1/chat/completions", json=body)
            assert response.status_code == status, (body, response.text)
            assert set(response.json()["error"]) == {"message", "type", "param", "code"}


def test_api_key_protects_v1_routes() -> None:
    client, _ = make_client(api_key="secret")
    with client:
        assert client.get("/v1/models").status_code == 401
        headers = {"Authorization": "Bearer secret"}
        assert client.get("/v1/models", headers=headers).status_code == 200
        assert client.get("/health").status_code == 200


def test_api_forwards_chat_template_kwargs() -> None:
    from fastapi.testclient import TestClient

    from dli_serve.api import create_app

    tokenizer = FakeTokenizer()
    app = create_app(
        Scheduler(FakeRunner(), tokenizer.decode), tokenizer, model_name="m", enable_thinking=True
    )
    with TestClient(app) as client:
        body = {"model": "m", "messages": [{"role": "user", "content": "Hi"}]}
        client.post("/v1/chat/completions", json=body)
        assert tokenizer.last_template_kwargs == {"enable_thinking": True}
        client.post(
            "/v1/chat/completions",
            json={**body, "chat_template_kwargs": {"enable_thinking": False}},
        )
        assert tokenizer.last_template_kwargs == {"enable_thinking": False}


if __name__ == "__main__":
    test_prefix_cache_keeps_longest_exact_prefix_with_lru_eviction()
    test_text_helpers_hold_back_partial_characters_and_stop_strings()
    test_scheduler_streams_reply_and_reports_usage()
    test_identical_prompt_reuses_the_whole_prompt_snapshot()
    test_next_chat_turn_reuses_the_history_snapshot()
    test_round_robin_interleaves_concurrent_requests()
    test_requests_beyond_max_num_seqs_wait_their_turn()
    test_stop_strings_and_length_limits()
    test_cancel_and_failures_only_affect_their_request()
    test_submit_validates_prompt_and_sampling()
    test_threaded_scheduler_serves_requests_and_shuts_down()
    if importlib.util.find_spec("fastapi") is None or importlib.util.find_spec("httpx") is None:
        print("skipping HTTP API tests: FastAPI or httpx is not installed")
    else:
        test_api_models_health_stats_and_ui()
        test_api_chat_completion_and_cached_usage_across_turns()
        test_api_chat_streaming_follows_the_openai_chunk_format()
        test_api_completions_plain_and_streaming()
        test_api_errors_use_the_openai_error_format()
        test_api_key_protects_v1_routes()
        test_api_forwards_chat_template_kwargs()
