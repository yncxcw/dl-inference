# Qwen3.5 Terminal Chatbot

This example runs the text backbone of `Qwen/Qwen3.5-0.8B` as a small terminal
chatbot. It uses the public Hugging Face `AutoTokenizer`,
`Qwen3_5ForCausalLM`, and `TextIteratorStreamer` APIs. Each turn is rendered
with the model's chat template, generated autoregressively, streamed as it is
decoded, and then appended to the conversation history.

Qwen3.5 is also a multimodal model family, but this example is deliberately
text-only. Image and video inputs require the conditional-generation model and
processor APIs instead.

## Install

Qwen3.5 support starts in Transformers 5.2. Upgrade older installations before
running the example:

```bash
python3 -m pip install --upgrade "transformers>=5.2.0" torch
```

The first run downloads the checkpoint from the Hugging Face Hub. A CPU works
but generates slowly; CUDA or MPS is selected automatically when available.

## Chat

```bash
python3 examples/qwen3_5/main.py
```

Use `/reset` to clear the conversation while preserving an optional system
prompt, or `/exit` to quit. The entire completed history is included when the
next turn is templated, so follow-up questions retain context.

Useful options include:

```bash
python3 examples/qwen3_5/main.py \
  --system-prompt "Answer briefly and accurately." \
  --max-new-tokens 256 \
  --temperature 1.0 \
  --top-p 1.0 \
  --top-k 20
```

Use `--temperature 0` for deterministic greedy decoding. Select a device
explicitly with `--device cpu`, `--device cuda`, or `--device mps`.
Non-thinking mode is the default and follows the checkpoint's chat template;
pass `--thinking` to opt in (the 0.8B model can loop in thinking mode).

## One-shot and offline use

Run one prompt, stream its answer, and exit:

```bash
python3 examples/qwen3_5/main.py --prompt "Explain KV caching in two sentences."
```

After the model is cached, prevent any Hub access with:

```bash
python3 examples/qwen3_5/main.py --local-files-only
```

You can also pass a local checkpoint directory to `--model-id`.

## Design notes

For every user turn, the session:

1. adds the user message to its in-memory history;
2. calls `tokenizer.apply_chat_template(..., add_generation_prompt=True)`;
3. starts `model.generate()` on a worker thread;
4. prints decoded chunks from `TextIteratorStreamer`; and
5. commits the complete assistant message to history.

If generation fails, the pending user turn is rolled back so the next request
does not inherit a half-completed exchange. Transformers manages the model's
per-generation KV cache. Before generation, the session verifies that the
rendered prompt plus `--max-new-tokens` fits `max_position_embeddings`; an
oversized turn is rejected without changing history. The example intentionally
re-renders full history on each turn; a production service should also choose a
history truncation or summarization policy, isolate sessions, add cancellation,
and bound concurrent generation.

Generation stops on both Qwen3.5 termination markers: the model EOS token
(`<|endoftext|>`, 248044) and the chat turn terminator (`<|im_end|>`, 248046).

## Offline tests

The tests use fake tokenizer, model, and streamer objects. They do not download
or instantiate model weights:

```bash
python3 -m unittest discover -s examples/qwen3_5 -p "test_*.py" -v
```
