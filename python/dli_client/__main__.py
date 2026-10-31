"""Terminal chat against a server: `python -m dli_client --base-url URL`."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from . import APIError, Client


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="dli-chat", description="Chat with an OpenAI-compatible server from the terminal."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key")
    parser.add_argument("--model", help="model name (default: the server's first model)")
    parser.add_argument("--system", help="system prompt")
    parser.add_argument("--prompt", help="send one message and exit")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--thinking", action="store_true", help="enable Qwen thinking mode")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    client = Client(base_url=args.base_url, api_key=args.api_key)
    try:
        model = args.model or client.models.list().data[0].id
    except APIError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    messages: list[dict[str, str]] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    def reply(text: str) -> None:
        messages.append({"role": "user", "content": text})
        answer = []
        usage = None
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            chat_template_kwargs={"enable_thinking": args.thinking},
        )
        try:
            for chunk in stream:
                usage = chunk.usage or usage
                for choice in chunk.choices:
                    text_delta = choice.delta.content or ""
                    answer.append(text_delta)
                    print(text_delta, end="", flush=True)
        except BaseException:
            messages.pop()
            raise
        print()
        messages.append({"role": "assistant", "content": "".join(answer)})
        if usage is not None:
            details = usage.prompt_tokens_details
            cached = details.cached_tokens if details is not None else 0
            print(
                f"[{usage.completion_tokens} tokens, {cached}/{usage.prompt_tokens} "
                "prompt tokens cached]",
                file=sys.stderr,
            )

    try:
        if args.prompt is not None:
            reply(args.prompt)
            return 0
        print(f"Chatting with {model}. Commands: /reset clears the conversation; /exit quits.")
        while True:
            try:
                text = input("You: ").strip()
            except EOFError:
                print()
                return 0
            if text == "/exit":
                return 0
            if text == "/reset":
                del messages[1 if args.system else 0 :]
                print("Conversation reset.")
                continue
            if text:
                print("Assistant: ", end="", flush=True)
                try:
                    reply(text)
                except KeyboardInterrupt:
                    print(" [interrupted]")
    except APIError as error:
        print(f"\nerror: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print()
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
