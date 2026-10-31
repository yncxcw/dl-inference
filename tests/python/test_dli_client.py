from __future__ import annotations

import importlib.util
import socket
import threading
import time

from serve_fakes import ASSISTANT, FakeRunner, FakeTokenizer


class LiveServer:
    """Runs the app on an ephemeral localhost port in a background thread."""

    def __init__(self, *, step_delay: float = 0.0, **app_kwargs) -> None:
        self.step_delay = step_delay
        self.app_kwargs = app_kwargs

    def __enter__(self) -> LiveServer:
        import uvicorn

        from dli_serve.api import create_app
        from dli_serve.scheduler import Scheduler

        self.runner = FakeRunner(step_delay=self.step_delay)
        scheduler = Scheduler(self.runner, FakeTokenizer().decode)
        app = create_app(scheduler, FakeTokenizer(), model_name="fake-model", **self.app_kwargs)
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        self.base_url = f"http://127.0.0.1:{sock.getsockname()[1]}/v1"
        self.server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
        self.thread = threading.Thread(
            target=self.server.run, kwargs={"sockets": [sock]}, daemon=True
        )
        self.thread.start()
        deadline = time.monotonic() + 10
        while not self.server.started:
            if time.monotonic() > deadline or not self.thread.is_alive():
                raise RuntimeError("the test server did not start")
            time.sleep(0.01)
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.server.should_exit = True
        self.thread.join(10)


HI = [{"role": "user", "content": "Hi"}]


def test_client_chat_completions_models_and_streaming() -> None:
    from dli_client import Client

    with LiveServer() as server:
        client = Client(base_url=server.base_url)
        models = client.models.list()
        assert [model.id for model in models.data] == ["fake-model"]
        assert client.models.retrieve("fake-model").max_model_len == 512

        reply = client.chat.completions.create(model="fake-model", messages=HI, temperature=0)
        assert reply.choices[0].message.content == "Hello there!"
        assert reply.choices[0].finish_reason == "stop"
        assert reply.usage.total_tokens == reply.usage.prompt_tokens + 13
        assert reply.not_a_field is None
        assert reply.to_dict()["object"] == "chat.completion"

        with client.chat.completions.create(
            model="fake-model",
            messages=HI,
            stream=True,
            temperature=0,
            stream_options={"include_usage": True},
        ) as stream:
            chunks = list(stream)
        text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
        assert text == "Hello there!"
        assert chunks[-1].usage.prompt_tokens_details.cached_tokens == reply.usage.prompt_tokens


def test_client_completions_and_errors() -> None:
    from dli_client import APIConnectionError, APIStatusError, Client

    with LiveServer(api_key="secret") as server:
        client = Client(base_url=server.base_url, api_key="secret")
        completion = client.completions.create(
            model="fake-model", prompt="Say:" + chr(ASSISTANT), max_tokens=5, temperature=0
        )
        assert completion.choices[0].text == "Hello"
        assert completion.choices[0].finish_reason == "length"

        streamed = client.completions.create(
            model="fake-model", prompt=FakeTokenizer().encode("Go:") + [ASSISTANT], stream=True
        )
        assert "".join(chunk.choices[0].text for chunk in streamed) == "Hello there!"

        for bad_client, status in (
            (Client(base_url=server.base_url), 401),
            (client, 404),
        ):
            try:
                bad_client.chat.completions.create(model="missing", messages=HI)
            except APIStatusError as error:
                assert error.status_code == status and error.message
            else:
                raise AssertionError("an invalid request succeeded")
        port = server.base_url.split(":")[2].split("/")[0]

    try:
        Client(base_url=f"http://127.0.0.1:{port}/v1", timeout=2).models.list()
    except APIConnectionError:
        pass
    else:
        raise AssertionError("a stopped server answered")


def test_closing_a_stream_cancels_generation_on_the_server() -> None:
    from dli_client import Client

    with LiveServer(step_delay=0.05) as server:
        client = Client(base_url=server.base_url)
        stream = client.chat.completions.create(model="fake-model", messages=HI, stream=True)
        next(iter(stream))  # the role chunk arrives before any model step finishes
        stream.close()
        deadline = time.monotonic() + 10
        stats = {}
        while time.monotonic() < deadline:
            import json
            import urllib.request

            with urllib.request.urlopen(server.base_url.removesuffix("/v1") + "/stats") as response:
                stats = json.loads(response.read())
            if stats["aborted"] == 1:
                break
            time.sleep(0.05)
        assert stats["aborted"] == 1 and stats["finished"] == 0, stats
        # Far fewer steps than a full reply: the server stopped early.
        assert len(server.runner.steps) < len(FakeTokenizer().encode("<user>Hi\n")) + 13


def test_official_openai_sdk_is_compatible() -> None:
    import openai

    with LiveServer() as server:
        client = openai.OpenAI(base_url=server.base_url, api_key="unused")
        assert [model.id for model in client.models.list()] == ["fake-model"]
        reply = client.chat.completions.create(model="fake-model", messages=HI, temperature=0)
        assert reply.choices[0].message.content == "Hello there!"
        assert reply.usage.prompt_tokens_details.cached_tokens == 0
        stream = client.chat.completions.create(
            model="fake-model", messages=HI, temperature=0, stream=True
        )
        assert "".join(c.choices[0].delta.content or "" for c in stream if c.choices) == (
            "Hello there!"
        )
        try:
            client.chat.completions.create(model="fake-model", messages=HI, n=2)
        except openai.BadRequestError as error:
            assert "n=1" in str(error)
        else:
            raise AssertionError("n=2 was accepted")


if __name__ == "__main__":
    if importlib.util.find_spec("fastapi") is None or importlib.util.find_spec("uvicorn") is None:
        print("skipping client tests: FastAPI or uvicorn is not installed")
        raise SystemExit(0)
    test_client_chat_completions_models_and_streaming()
    test_client_completions_and_errors()
    test_closing_a_stream_cancels_generation_on_the_server()
    if importlib.util.find_spec("openai") is None:
        print("skipping the openai SDK compatibility test: openai is not installed")
    else:
        test_official_openai_sdk_is_compatible()
